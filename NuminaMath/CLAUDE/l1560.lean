import Mathlib

namespace NUMINAMATH_CALUDE_ratio_of_segments_l1560_156041

/-- Given points A, B, C, and D on a line in that order, with AB : AC = 1 : 5 and BC : CD = 2 : 1, prove AB : CD = 1 : 2 -/
theorem ratio_of_segments (A B C D : ℝ) (h_order : A < B ∧ B < C ∧ C < D) 
  (h_ratio1 : (B - A) / (C - A) = 1 / 5)
  (h_ratio2 : (C - B) / (D - C) = 2 / 1) :
  (B - A) / (D - C) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_segments_l1560_156041


namespace NUMINAMATH_CALUDE_inequality_proof_l1560_156030

theorem inequality_proof (n : ℕ) (h : n > 2) :
  (2*n - 1)^n + (2*n)^n < (2*n + 1)^n := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1560_156030


namespace NUMINAMATH_CALUDE_sphere_volume_from_cylinder_volume_l1560_156082

/-- The volume of a sphere with the same radius as a cylinder of volume 72π -/
theorem sphere_volume_from_cylinder_volume (r : ℝ) (h : ℝ) :
  (π * r^2 * h = 72 * π) →
  ((4 / 3) * π * r^3 = 48 * π) :=
by sorry

end NUMINAMATH_CALUDE_sphere_volume_from_cylinder_volume_l1560_156082


namespace NUMINAMATH_CALUDE_robin_extra_gum_l1560_156018

/-- Represents the number of extra pieces of gum Robin has -/
def extra_gum (total_pieces packages pieces_per_package : ℕ) : ℕ :=
  total_pieces - packages * pieces_per_package

/-- Proves that Robin has 6 extra pieces of gum given the conditions -/
theorem robin_extra_gum :
  extra_gum 41 5 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_robin_extra_gum_l1560_156018


namespace NUMINAMATH_CALUDE_license_plate_count_l1560_156055

/-- The number of possible letters in each position of the license plate -/
def num_letters : ℕ := 26

/-- The number of possible digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of possible even digits -/
def num_even_digits : ℕ := 5

/-- The number of possible odd digits -/
def num_odd_digits : ℕ := 5

/-- The total number of license plates with 3 letters followed by 2 digits,
    where one digit is odd and the other is even -/
def total_license_plates : ℕ := num_letters^3 * num_digits * num_even_digits

theorem license_plate_count :
  total_license_plates = 878800 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l1560_156055


namespace NUMINAMATH_CALUDE_optimal_tank_design_l1560_156047

/-- Represents the dimensions and cost of a rectangular open-top water storage tank. -/
structure Tank where
  length : ℝ
  width : ℝ
  depth : ℝ
  base_cost : ℝ
  wall_cost : ℝ

/-- Calculates the volume of the tank. -/
def volume (t : Tank) : ℝ := t.length * t.width * t.depth

/-- Calculates the total construction cost of the tank. -/
def construction_cost (t : Tank) : ℝ :=
  t.base_cost * t.length * t.width + t.wall_cost * 2 * (t.length + t.width) * t.depth

/-- Theorem stating the optimal dimensions and minimum cost for the tank. -/
theorem optimal_tank_design :
  ∃ (t : Tank),
    t.depth = 3 ∧
    volume t = 4800 ∧
    t.base_cost = 150 ∧
    t.wall_cost = 120 ∧
    t.length = t.width ∧
    t.length = 40 ∧
    construction_cost t = 297600 ∧
    ∀ (t' : Tank),
      t'.depth = 3 →
      volume t' = 4800 →
      t'.base_cost = 150 →
      t'.wall_cost = 120 →
      construction_cost t' ≥ construction_cost t :=
by sorry

end NUMINAMATH_CALUDE_optimal_tank_design_l1560_156047


namespace NUMINAMATH_CALUDE_two_colonies_reach_limit_same_time_l1560_156040

/-- Represents the number of days it takes for a bacteria colony to reach its habitat limit -/
def habitat_limit_days : ℕ := 22

/-- Represents the daily growth factor of the bacteria colony -/
def daily_growth_factor : ℕ := 2

/-- Theorem stating that two bacteria colonies starting simultaneously will reach the habitat limit in the same number of days as a single colony -/
theorem two_colonies_reach_limit_same_time (initial_population : ℕ) :
  (initial_population * daily_growth_factor ^ habitat_limit_days) =
  (2 * initial_population * daily_growth_factor ^ habitat_limit_days) / 2 :=
by
  sorry

#check two_colonies_reach_limit_same_time

end NUMINAMATH_CALUDE_two_colonies_reach_limit_same_time_l1560_156040


namespace NUMINAMATH_CALUDE_percentage_commutation_l1560_156084

theorem percentage_commutation (x : ℝ) : 
  (0.4 * (0.3 * x) = 24) → (0.3 * (0.4 * x) = 24) := by
  sorry

end NUMINAMATH_CALUDE_percentage_commutation_l1560_156084


namespace NUMINAMATH_CALUDE_monthly_donation_proof_l1560_156096

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- The total annual donation in dollars -/
def annual_donation : ℕ := 17436

/-- The monthly donation in dollars -/
def monthly_donation : ℕ := annual_donation / months_in_year

theorem monthly_donation_proof : monthly_donation = 1453 := by
  sorry

end NUMINAMATH_CALUDE_monthly_donation_proof_l1560_156096


namespace NUMINAMATH_CALUDE_average_speed_calculation_l1560_156001

theorem average_speed_calculation (speed1 speed2 : ℝ) (time : ℝ) 
  (h1 : speed1 = 20)
  (h2 : speed2 = 30)
  (h3 : time = 2) :
  (speed1 + speed2) / time = 25 :=
by sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l1560_156001


namespace NUMINAMATH_CALUDE_knicks_knacks_knocks_conversion_l1560_156083

/-- Given the conversions between knicks, knacks, and knocks, 
    prove that 80 knocks is equal to 192 knicks. -/
theorem knicks_knacks_knocks_conversion : 
  ∀ (knicks knacks knocks : ℚ),
    (9 * knicks = 3 * knacks) →
    (4 * knacks = 5 * knocks) →
    (80 * knocks = 192 * knicks) :=
by
  sorry

end NUMINAMATH_CALUDE_knicks_knacks_knocks_conversion_l1560_156083


namespace NUMINAMATH_CALUDE_zero_in_interval_l1560_156073

-- Define the function f(x) = x³ + x - 3
def f (x : ℝ) : ℝ := x^3 + x - 3

-- Theorem statement
theorem zero_in_interval : ∃ x : ℝ, x > 1 ∧ x < 2 ∧ f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_in_interval_l1560_156073


namespace NUMINAMATH_CALUDE_unique_solution_cube_difference_square_l1560_156094

theorem unique_solution_cube_difference_square (x y z : ℕ+) : 
  (x.val : ℤ)^3 - (y.val : ℤ)^3 = (z.val : ℤ)^2 →
  Nat.Prime y.val →
  ¬(3 ∣ z.val) →
  ¬(y.val ∣ z.val) →
  x = 8 ∧ y = 7 ∧ z = 13 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_cube_difference_square_l1560_156094


namespace NUMINAMATH_CALUDE_jason_average_messages_l1560_156061

/-- The average number of text messages sent over five days -/
def average_messages (monday : ℕ) (tuesday : ℕ) (wed_to_fri : ℕ) (days : ℕ) : ℚ :=
  (monday + tuesday + 3 * wed_to_fri : ℚ) / days

theorem jason_average_messages :
  let monday := 220
  let tuesday := monday / 2
  let wed_to_fri := 50
  let days := 5
  average_messages monday tuesday wed_to_fri days = 96 := by
sorry

end NUMINAMATH_CALUDE_jason_average_messages_l1560_156061


namespace NUMINAMATH_CALUDE_exists_line_through_P_intersecting_hyperbola_l1560_156038

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/2 = 1

-- Define the point P
def P : ℝ × ℝ := (2, 1)

-- Define a line passing through P with slope k
def line (k : ℝ) (x y : ℝ) : Prop := y - P.2 = k * (x - P.1)

-- Define the midpoint condition
def is_midpoint (p a b : ℝ × ℝ) : Prop :=
  p.1 = (a.1 + b.1) / 2 ∧ p.2 = (a.2 + b.2) / 2

-- Theorem statement
theorem exists_line_through_P_intersecting_hyperbola :
  ∃ (k : ℝ) (A B : ℝ × ℝ),
    hyperbola A.1 A.2 ∧
    hyperbola B.1 B.2 ∧
    line k A.1 A.2 ∧
    line k B.1 B.2 ∧
    is_midpoint P A B :=
  sorry

end NUMINAMATH_CALUDE_exists_line_through_P_intersecting_hyperbola_l1560_156038


namespace NUMINAMATH_CALUDE_geometric_progression_sufficient_not_necessary_l1560_156032

/-- A sequence of three real numbers forms a geometric progression --/
def is_geometric_progression (a b c : ℝ) : Prop :=
  ∃ r : ℝ, b = a * r ∧ c = b * r

theorem geometric_progression_sufficient_not_necessary :
  (∀ a b c : ℝ, is_geometric_progression a b c → b^2 = a*c) ∧
  (∃ a b c : ℝ, b^2 = a*c ∧ ¬is_geometric_progression a b c) := by
  sorry


end NUMINAMATH_CALUDE_geometric_progression_sufficient_not_necessary_l1560_156032


namespace NUMINAMATH_CALUDE_max_sum_given_constraints_l1560_156057

theorem max_sum_given_constraints (x y : ℝ) (h1 : x^2 + y^2 = 100) (h2 : x * y = 40) :
  x + y ≤ 6 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_max_sum_given_constraints_l1560_156057


namespace NUMINAMATH_CALUDE_abs_2y_minus_7_zero_l1560_156004

theorem abs_2y_minus_7_zero (y : ℚ) : |2 * y - 7| = 0 ↔ y = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_abs_2y_minus_7_zero_l1560_156004


namespace NUMINAMATH_CALUDE_john_sells_20_woodburnings_l1560_156070

/-- The number of woodburnings John sells -/
def num_woodburnings : ℕ := 20

/-- The selling price of each woodburning in dollars -/
def selling_price : ℕ := 15

/-- The cost of wood in dollars -/
def wood_cost : ℕ := 100

/-- John's profit in dollars -/
def profit : ℕ := 200

/-- Theorem stating that the number of woodburnings John sells is 20 -/
theorem john_sells_20_woodburnings :
  num_woodburnings = 20 ∧
  selling_price * num_woodburnings = wood_cost + profit := by
  sorry

end NUMINAMATH_CALUDE_john_sells_20_woodburnings_l1560_156070


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1560_156002

/-- An arithmetic sequence {a_n} where a_1 = 1/3, a_2 + a_5 = 4, and a_n = 33 has n = 50 -/
theorem arithmetic_sequence_problem (a : ℕ → ℚ) (n : ℕ) 
  (h_arith : ∀ k, a (k + 1) - a k = a (k + 2) - a (k + 1)) 
  (h_a1 : a 1 = 1/3)
  (h_sum : a 2 + a 5 = 4)
  (h_an : a n = 33) :
  n = 50 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1560_156002


namespace NUMINAMATH_CALUDE_quadratic_integer_roots_count_l1560_156022

theorem quadratic_integer_roots_count : 
  let f (m : ℤ) := (∃ x₁ x₂ : ℤ, x₁^2 - m*x₁ + 36 = 0 ∧ x₂^2 - m*x₂ + 36 = 0 ∧ x₁ ≠ x₂)
  (∃! (s : Finset ℤ), (∀ m ∈ s, f m) ∧ s.card = 10) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_integer_roots_count_l1560_156022


namespace NUMINAMATH_CALUDE_equation_solution_l1560_156062

theorem equation_solution (x : ℝ) (hx1 : x ≠ 1) (hx2 : x ≠ -1) :
  (x / (x - 1) = 4 / (x^2 - 1) + 1) ↔ (x = 3) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l1560_156062


namespace NUMINAMATH_CALUDE_range_equivalence_l1560_156049

/-- The range of real numbers a for which at least one of the given equations has real roots -/
def range_with_real_roots (a : ℝ) : Prop :=
  ∃ x : ℝ, (x^2 + 4*a*x - 4*a + 3 = 0) ∨ 
            (x^2 + (a-1)*x + a^2 = 0) ∨ 
            (x^2 + 2*a*x - 2*a = 0)

/-- The range of real numbers a for which none of the given equations have real roots -/
def range_without_real_roots (a : ℝ) : Prop :=
  ∀ x : ℝ, (x^2 + 4*a*x - 4*a + 3 ≠ 0) ∧ 
            (x^2 + (a-1)*x + a^2 ≠ 0) ∧ 
            (x^2 + 2*a*x - 2*a ≠ 0)

/-- The theorem stating that the range with real roots is the complement of the range without real roots -/
theorem range_equivalence : 
  ∀ a : ℝ, range_with_real_roots a ↔ ¬(range_without_real_roots a) :=
sorry

end NUMINAMATH_CALUDE_range_equivalence_l1560_156049


namespace NUMINAMATH_CALUDE_evaluate_expression_l1560_156037

theorem evaluate_expression : 3000 * (3000^1500 + 3000^1500) = 2 * 3000^1501 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1560_156037


namespace NUMINAMATH_CALUDE_scientists_born_in_july_percentage_l1560_156019

theorem scientists_born_in_july_percentage :
  let total_scientists : ℕ := 120
  let born_in_july : ℕ := 20
  let percentage : ℚ := (born_in_july : ℚ) / total_scientists * 100
  percentage = 50 / 3 := by sorry

end NUMINAMATH_CALUDE_scientists_born_in_july_percentage_l1560_156019


namespace NUMINAMATH_CALUDE_joint_purchase_popularity_l1560_156031

structure JointPurchase where
  scale : ℝ
  cost_savings : ℝ
  quality_assessment : ℝ
  community_trust : ℝ
  transaction_costs : ℝ
  organizational_efforts : ℝ
  convenience : ℝ
  dispute_potential : ℝ

def benefits (jp : JointPurchase) : ℝ :=
  jp.cost_savings + jp.quality_assessment + jp.community_trust

def drawbacks (jp : JointPurchase) : ℝ :=
  jp.transaction_costs + jp.organizational_efforts + jp.convenience + jp.dispute_potential

theorem joint_purchase_popularity (jp : JointPurchase) :
  jp.scale > 1 → benefits jp > drawbacks jp ∧
  jp.scale ≤ 1 → benefits jp ≤ drawbacks jp :=
sorry

end NUMINAMATH_CALUDE_joint_purchase_popularity_l1560_156031


namespace NUMINAMATH_CALUDE_B_power_60_is_identity_l1560_156054

def B : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![1, 0, 0],
    ![0, 0, -1],
    ![0, 1, 0]]

theorem B_power_60_is_identity :
  B ^ 60 = 1 := by sorry

end NUMINAMATH_CALUDE_B_power_60_is_identity_l1560_156054


namespace NUMINAMATH_CALUDE_concert_attendance_l1560_156045

theorem concert_attendance (total_tickets : ℕ) 
  (h1 : total_tickets = 2465)
  (before_start : ℕ) 
  (h2 : before_start = (7 * total_tickets) / 8)
  (after_first_song : ℕ) 
  (h3 : after_first_song = (13 * (total_tickets - before_start)) / 17)
  (last_performances : ℕ) 
  (h4 : last_performances = 47) : 
  total_tickets - before_start - after_first_song - last_performances = 26 := by
sorry

end NUMINAMATH_CALUDE_concert_attendance_l1560_156045


namespace NUMINAMATH_CALUDE_union_equality_implies_t_value_l1560_156011

def M (t : ℝ) : Set ℝ := {1, 3, t}
def N (t : ℝ) : Set ℝ := {t^2 - t + 1}

theorem union_equality_implies_t_value (t : ℝ) :
  M t ∪ N t = M t → t = 0 ∨ t = 2 ∨ t = -1 := by
  sorry

end NUMINAMATH_CALUDE_union_equality_implies_t_value_l1560_156011


namespace NUMINAMATH_CALUDE_tank_fraction_problem_l1560_156093

/-- The problem of determining the fraction of the first tank's capacity that is filled. -/
theorem tank_fraction_problem (tank1_capacity tank2_capacity tank3_capacity : ℚ)
  (tank2_fill_fraction tank3_fill_fraction : ℚ)
  (total_water : ℚ) :
  tank1_capacity = 7000 →
  tank2_capacity = 5000 →
  tank3_capacity = 3000 →
  tank2_fill_fraction = 4/5 →
  tank3_fill_fraction = 1/2 →
  total_water = 10850 →
  total_water = tank1_capacity * (107/140) + tank2_capacity * tank2_fill_fraction + tank3_capacity * tank3_fill_fraction :=
by sorry

end NUMINAMATH_CALUDE_tank_fraction_problem_l1560_156093


namespace NUMINAMATH_CALUDE_linear_relationship_l1560_156048

/-- Given a linear relationship where an increase of 4 units in x corresponds to an increase of 6 units in y,
    prove that an increase of 12 units in x will result in an increase of 18 units in y. -/
theorem linear_relationship (f : ℝ → ℝ) (x₀ : ℝ) :
  (f (x₀ + 4) - f x₀ = 6) → (f (x₀ + 12) - f x₀ = 18) := by
  sorry

end NUMINAMATH_CALUDE_linear_relationship_l1560_156048


namespace NUMINAMATH_CALUDE_haleys_extra_tickets_l1560_156017

theorem haleys_extra_tickets (ticket_price : ℕ) (initial_tickets : ℕ) (total_spent : ℕ) : 
  ticket_price = 4 →
  initial_tickets = 3 →
  total_spent = 32 →
  total_spent / ticket_price - initial_tickets = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_haleys_extra_tickets_l1560_156017


namespace NUMINAMATH_CALUDE_compound_oxygen_count_l1560_156014

/-- Represents a chemical compound with a given number of Carbon, Hydrogen, and Oxygen atoms -/
structure Compound where
  carbon : ℕ
  hydrogen : ℕ
  oxygen : ℕ

/-- Calculates the molecular weight of a compound given atomic weights -/
def molecularWeight (c : Compound) (carbonWeight hydrogenWeight oxygenWeight : ℝ) : ℝ :=
  c.carbon * carbonWeight + c.hydrogen * hydrogenWeight + c.oxygen * oxygenWeight

/-- Theorem: A compound with 4 Carbon atoms, 1 Hydrogen atom, and a molecular weight of 65 g/mol contains 1 Oxygen atom -/
theorem compound_oxygen_count :
  ∃ (c : Compound),
    c.carbon = 4 ∧
    c.hydrogen = 1 ∧
    c.oxygen = 1 ∧
    molecularWeight c 12.01 1.008 16.00 = 65 := by
  sorry


end NUMINAMATH_CALUDE_compound_oxygen_count_l1560_156014


namespace NUMINAMATH_CALUDE_kids_difference_l1560_156053

theorem kids_difference (camp_kids home_kids : ℕ) 
  (h1 : camp_kids = 202958) 
  (h2 : home_kids = 777622) : 
  home_kids - camp_kids = 574664 := by
sorry

end NUMINAMATH_CALUDE_kids_difference_l1560_156053


namespace NUMINAMATH_CALUDE_car_production_is_four_l1560_156052

/-- Represents the factory's production and profit data -/
structure FactoryData where
  car_material_cost : ℕ
  car_selling_price : ℕ
  motorcycle_material_cost : ℕ
  motorcycle_count : ℕ
  motorcycle_selling_price : ℕ
  profit_difference : ℕ

/-- Calculates the number of cars that could be produced per month -/
def calculate_car_production (data : FactoryData) : ℕ :=
  let motorcycle_profit := data.motorcycle_count * data.motorcycle_selling_price - data.motorcycle_material_cost
  let car_profit := fun c => c * data.car_selling_price - data.car_material_cost
  (motorcycle_profit - data.profit_difference + data.car_material_cost) / data.car_selling_price

theorem car_production_is_four (data : FactoryData) 
  (h1 : data.car_material_cost = 100)
  (h2 : data.car_selling_price = 50)
  (h3 : data.motorcycle_material_cost = 250)
  (h4 : data.motorcycle_count = 8)
  (h5 : data.motorcycle_selling_price = 50)
  (h6 : data.profit_difference = 50) :
  calculate_car_production data = 4 := by
  sorry

end NUMINAMATH_CALUDE_car_production_is_four_l1560_156052


namespace NUMINAMATH_CALUDE_siblings_total_age_l1560_156026

/-- Represents the ages of six siblings -/
structure SiblingAges where
  susan : ℕ
  arthur : ℕ
  tom : ℕ
  bob : ℕ
  emily : ℕ
  david : ℕ

/-- Calculates the total age of all siblings -/
def totalAge (ages : SiblingAges) : ℕ :=
  ages.susan + ages.arthur + ages.tom + ages.bob + ages.emily + ages.david

/-- Theorem stating the total age of the siblings -/
theorem siblings_total_age :
  ∀ (ages : SiblingAges),
    ages.susan = 15 →
    ages.bob = 11 →
    ages.arthur = ages.susan + 2 →
    ages.tom = ages.bob - 3 →
    ages.emily = ages.susan / 2 →
    ages.david = (ages.arthur + ages.tom + ages.emily) / 3 →
    totalAge ages = 70 := by
  sorry


end NUMINAMATH_CALUDE_siblings_total_age_l1560_156026


namespace NUMINAMATH_CALUDE_binomial_prob_theorem_l1560_156086

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The variance of a binomial distribution -/
def variance (X : BinomialRV) : ℝ := X.n * X.p * (1 - X.p)

/-- The probability mass function of a binomial distribution -/
def pmf (X : BinomialRV) (k : ℕ) : ℝ :=
  (Nat.choose X.n k) * (X.p ^ k) * ((1 - X.p) ^ (X.n - k))

/-- Theorem: If X ~ B(10,p) with D(X) = 2.4 and P(X=4) > P(X=6), then p = 0.4 -/
theorem binomial_prob_theorem (X : BinomialRV) 
  (h_n : X.n = 10)
  (h_var : variance X = 2.4)
  (h_prob : pmf X 4 > pmf X 6) :
  X.p = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_binomial_prob_theorem_l1560_156086


namespace NUMINAMATH_CALUDE_certain_number_proof_l1560_156088

theorem certain_number_proof : ∃ n : ℕ, n = 213 * 16 ∧ n = 3408 := by
  -- Given condition: 0.016 * 2.13 = 0.03408
  have h : (0.016 : ℝ) * 2.13 = 0.03408 := by sorry
  
  -- Proof that 213 * 16 = 3408
  sorry


end NUMINAMATH_CALUDE_certain_number_proof_l1560_156088


namespace NUMINAMATH_CALUDE_base_four_20314_equals_568_l1560_156029

def base_four_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

theorem base_four_20314_equals_568 :
  base_four_to_decimal [4, 1, 3, 0, 2] = 568 := by
  sorry

end NUMINAMATH_CALUDE_base_four_20314_equals_568_l1560_156029


namespace NUMINAMATH_CALUDE_train_length_l1560_156078

theorem train_length (bridge_length : ℝ) (crossing_time : ℝ) (train_speed : ℝ) :
  bridge_length = 300 →
  crossing_time = 45 →
  train_speed = 47.99999999999999 →
  (train_speed * crossing_time) - bridge_length = 1860 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l1560_156078


namespace NUMINAMATH_CALUDE_tobys_friends_l1560_156006

theorem tobys_friends (total_friends : ℕ) (boy_friends : ℕ) (girl_friends : ℕ) : 
  (boy_friends : ℚ) / total_friends = 55 / 100 →
  boy_friends = 33 →
  girl_friends = 27 :=
by sorry

end NUMINAMATH_CALUDE_tobys_friends_l1560_156006


namespace NUMINAMATH_CALUDE_postcard_selling_price_l1560_156092

/-- Proves that the selling price per postcard is $10 --/
theorem postcard_selling_price 
  (initial_postcards : ℕ)
  (sold_postcards : ℕ)
  (new_postcard_price : ℚ)
  (final_postcard_count : ℕ)
  (h1 : initial_postcards = 18)
  (h2 : sold_postcards = initial_postcards / 2)
  (h3 : new_postcard_price = 5)
  (h4 : final_postcard_count = 36)
  : (sold_postcards : ℚ) * (final_postcard_count - initial_postcards) * new_postcard_price / sold_postcards = 10 := by
  sorry

end NUMINAMATH_CALUDE_postcard_selling_price_l1560_156092


namespace NUMINAMATH_CALUDE_triangle_altitude_length_l1560_156067

theorem triangle_altitude_length (r : ℝ) (h : r > 0) : 
  let square_side : ℝ := 4 * r
  let square_area : ℝ := square_side ^ 2
  let diagonal_length : ℝ := square_side * Real.sqrt 2
  let triangle_area : ℝ := 2 * square_area
  let altitude : ℝ := 2 * triangle_area / diagonal_length
  altitude = 8 * r * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_triangle_altitude_length_l1560_156067


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1560_156050

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  (∀ n : ℕ, a n > 0) →
  a 1 + a 2 = 4/9 →
  a 3 + a 4 + a 5 + a 6 = 40 →
  (a 7 + a 8 + a 9) / 9 = 117 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1560_156050


namespace NUMINAMATH_CALUDE_four_solutions_l1560_156008

def is_solution (m n : ℕ) : Prop :=
  m > 0 ∧ n > 0 ∧ (4 : ℚ) / m + (2 : ℚ) / n = 1

def solution_count : ℕ := 4

theorem four_solutions :
  ∃ (S : Finset (ℕ × ℕ)), S.card = solution_count ∧
    (∀ (p : ℕ × ℕ), p ∈ S ↔ is_solution p.1 p.2) :=
sorry

end NUMINAMATH_CALUDE_four_solutions_l1560_156008


namespace NUMINAMATH_CALUDE_nonagon_diagonals_l1560_156007

/-- The number of distinct diagonals in a convex nonagon -/
def diagonals_in_nonagon : ℕ := 27

/-- A convex nonagon has 27 distinct diagonals -/
theorem nonagon_diagonals : diagonals_in_nonagon = 27 := by sorry

end NUMINAMATH_CALUDE_nonagon_diagonals_l1560_156007


namespace NUMINAMATH_CALUDE_inequality_solution_l1560_156081

theorem inequality_solution (x : ℝ) : 
  (5 * x^2 + 10 * x - 34) / ((x - 2) * (3 * x + 5)) < 2 ↔ 
  x < -5/3 ∨ x > 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l1560_156081


namespace NUMINAMATH_CALUDE_n2o3_molecular_weight_l1560_156021

/-- The atomic weight of nitrogen in g/mol -/
def nitrogen_weight : ℝ := 14.01

/-- The atomic weight of oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The molecular weight of N2O3 in g/mol -/
def n2o3_weight : ℝ := 2 * nitrogen_weight + 3 * oxygen_weight

/-- Theorem stating that the molecular weight of N2O3 is 76.02 g/mol -/
theorem n2o3_molecular_weight : n2o3_weight = 76.02 := by
  sorry

end NUMINAMATH_CALUDE_n2o3_molecular_weight_l1560_156021


namespace NUMINAMATH_CALUDE_original_amount_is_1160_l1560_156099

/-- Given an initial principal, time period, and interest rates, calculate the final amount using simple interest. -/
def simple_interest_amount (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Prove that under the given conditions, the amount after 3 years at the original interest rate is $1160. -/
theorem original_amount_is_1160 
  (principal : ℝ) 
  (original_rate : ℝ) 
  (time : ℝ) 
  (h_principal : principal = 800) 
  (h_time : time = 3) 
  (h_increased_amount : simple_interest_amount principal (original_rate + 0.03) time = 992) :
  simple_interest_amount principal original_rate time = 1160 := by
sorry

end NUMINAMATH_CALUDE_original_amount_is_1160_l1560_156099


namespace NUMINAMATH_CALUDE_range_of_fraction_l1560_156076

theorem range_of_fraction (a b : ℝ) 
  (ha : 0 < a ∧ a ≤ 2) 
  (hb : b ≥ 1)
  (hba : b ≤ a^2) :
  ∃ (t : ℝ), t = b / a ∧ 1/2 ≤ t ∧ t ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_fraction_l1560_156076


namespace NUMINAMATH_CALUDE_tangent_perpendicular_and_inequality_l1560_156046

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := ((4*x + a) * Real.log x) / (3*x + 1)

theorem tangent_perpendicular_and_inequality (a : ℝ) :
  (∃ m : ℝ, ∀ x : ℝ, x ≥ 1 → f a x ≤ m * (x - 1)) →
  (a = 0 ∧ ∀ m : ℝ, (∀ x : ℝ, x ≥ 1 → f a x ≤ m * (x - 1)) → m ≥ 1) :=
sorry

end NUMINAMATH_CALUDE_tangent_perpendicular_and_inequality_l1560_156046


namespace NUMINAMATH_CALUDE_unique_solution_system_l1560_156090

theorem unique_solution_system (x y : ℚ) : 
  (3 * x + 2 * y = 7 ∧ 6 * x - 5 * y = 4) ↔ (x = 43/27 ∧ y = 10/9) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_system_l1560_156090


namespace NUMINAMATH_CALUDE_xy_problem_l1560_156042

theorem xy_problem (x y : ℝ) (h1 : x * y = 9) (h2 : x / y = 36) (hx : x > 0) (hy : y > 0) : y = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_xy_problem_l1560_156042


namespace NUMINAMATH_CALUDE_fourth_boy_payment_l1560_156098

theorem fourth_boy_payment (a b c d : ℝ) : 
  a + b + c + d = 80 →
  a = (1/2) * (b + c + d) →
  b = (1/4) * (a + c + d) →
  c = (1/3) * (a + b + d) →
  d + 5 = 23 := by
  sorry

end NUMINAMATH_CALUDE_fourth_boy_payment_l1560_156098


namespace NUMINAMATH_CALUDE_quadratic_root_implies_m_value_l1560_156033

theorem quadratic_root_implies_m_value (m : ℝ) : 
  (1 : ℝ)^2 - 2*m*(1 : ℝ) + 1 = 0 → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_m_value_l1560_156033


namespace NUMINAMATH_CALUDE_roberta_record_listening_time_l1560_156077

theorem roberta_record_listening_time :
  let x : ℕ := 8  -- initial number of records
  let y : ℕ := 12 -- additional records received
  let z : ℕ := 30 -- records bought
  let t : ℕ := 2  -- time needed to listen to each record in days
  (x + y + z) * t = 100 := by sorry

end NUMINAMATH_CALUDE_roberta_record_listening_time_l1560_156077


namespace NUMINAMATH_CALUDE_average_weight_decrease_l1560_156058

/-- Proves that replacing a 72 kg student with a 12 kg student in a group of 5 decreases the average weight by 12 kg -/
theorem average_weight_decrease (initial_average : ℝ) : 
  let total_weight := 5 * initial_average
  let new_total_weight := total_weight - 72 + 12
  let new_average := new_total_weight / 5
  initial_average - new_average = 12 := by
sorry

end NUMINAMATH_CALUDE_average_weight_decrease_l1560_156058


namespace NUMINAMATH_CALUDE_units_digit_G_1000_l1560_156000

-- Define G_n
def G (n : ℕ) : ℕ := 3 * 2^(2^n) + 4

-- Theorem statement
theorem units_digit_G_1000 : G 1000 % 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_G_1000_l1560_156000


namespace NUMINAMATH_CALUDE_angle_bisector_length_l1560_156012

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the angle bisector of B
def angleBisector (t : Triangle) : ℝ × ℝ := sorry

-- Define the length of a line segment
def length (p q : ℝ × ℝ) : ℝ := sorry

-- Define the measure of an angle
def angleMeasure (p q r : ℝ × ℝ) : ℝ := sorry

theorem angle_bisector_length 
  (t : Triangle) 
  (h1 : angleMeasure t.A t.B t.C = 20)
  (h2 : angleMeasure t.C t.A t.B = 40)
  (h3 : length t.A t.C - length t.A t.B = 5) :
  length t.B (angleBisector t) = 5 := by sorry

end NUMINAMATH_CALUDE_angle_bisector_length_l1560_156012


namespace NUMINAMATH_CALUDE_min_green_surface_fraction_l1560_156091

/-- Represents a cube with given edge length -/
structure Cube where
  edge : ℕ
  deriving Repr

/-- Represents the composition of a large cube -/
structure CubeComposition where
  large_cube : Cube
  small_cube : Cube
  blue_count : ℕ
  green_count : ℕ
  deriving Repr

/-- Calculates the surface area of a cube -/
def surface_area (c : Cube) : ℕ := 6 * c.edge^2

/-- Calculates the volume of a cube -/
def volume (c : Cube) : ℕ := c.edge^3

/-- Theorem: Minimum green surface area fraction -/
theorem min_green_surface_fraction (cc : CubeComposition) 
  (h1 : cc.large_cube.edge = 4)
  (h2 : cc.small_cube.edge = 1)
  (h3 : volume cc.large_cube = cc.blue_count + cc.green_count)
  (h4 : cc.blue_count = 50)
  (h5 : cc.green_count = 14) :
  (6 : ℚ) / surface_area cc.large_cube = 1/16 := by
  sorry

end NUMINAMATH_CALUDE_min_green_surface_fraction_l1560_156091


namespace NUMINAMATH_CALUDE_unique_perfect_square_l1560_156080

theorem unique_perfect_square (n : ℕ+) : n^2 - 19*n - 99 = m^2 ↔ n = 199 :=
  sorry

end NUMINAMATH_CALUDE_unique_perfect_square_l1560_156080


namespace NUMINAMATH_CALUDE_kylies_coins_l1560_156069

/-- Kylie's coin collection problem -/
theorem kylies_coins (coins_from_piggy_bank coins_from_father coins_to_laura coins_left : ℕ) 
  (h1 : coins_from_piggy_bank = 15)
  (h2 : coins_from_father = 8)
  (h3 : coins_to_laura = 21)
  (h4 : coins_left = 15) :
  coins_from_piggy_bank + coins_from_father + coins_to_laura - coins_left = 13 := by
  sorry

#check kylies_coins

end NUMINAMATH_CALUDE_kylies_coins_l1560_156069


namespace NUMINAMATH_CALUDE_min_value_sqrt_expression_l1560_156095

theorem min_value_sqrt_expression (x : ℝ) (hx : x > 0) :
  4 * Real.sqrt x + 2 / Real.sqrt x ≥ 6 ∧
  ∃ y : ℝ, y > 0 ∧ 4 * Real.sqrt y + 2 / Real.sqrt y = 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sqrt_expression_l1560_156095


namespace NUMINAMATH_CALUDE_journey_speed_proof_l1560_156087

/-- Proves that given a journey of 336 km completed in 15 hours, where the second half is traveled at 24 km/hr, the speed for the first half of the journey is 21 km/hr. -/
theorem journey_speed_proof (total_distance : ℝ) (total_time : ℝ) (second_half_speed : ℝ) :
  total_distance = 336 →
  total_time = 15 →
  second_half_speed = 24 →
  let first_half_distance : ℝ := total_distance / 2
  let second_half_distance : ℝ := total_distance / 2
  let second_half_time : ℝ := second_half_distance / second_half_speed
  let first_half_time : ℝ := total_time - second_half_time
  let first_half_speed : ℝ := first_half_distance / first_half_time
  first_half_speed = 21 :=
by sorry

end NUMINAMATH_CALUDE_journey_speed_proof_l1560_156087


namespace NUMINAMATH_CALUDE_hen_count_l1560_156051

theorem hen_count (total_animals : ℕ) (total_feet : ℕ) (hen_feet : ℕ) (cow_feet : ℕ) 
  (h1 : total_animals = 48) 
  (h2 : total_feet = 136) 
  (h3 : hen_feet = 2) 
  (h4 : cow_feet = 4) :
  ∃ (hens cows : ℕ), 
    hens + cows = total_animals ∧ 
    hen_feet * hens + cow_feet * cows = total_feet ∧ 
    hens = 28 := by
  sorry

end NUMINAMATH_CALUDE_hen_count_l1560_156051


namespace NUMINAMATH_CALUDE_g_difference_l1560_156016

-- Define the sum of divisors function
def sigma (n : ℕ) : ℕ := sorry

-- Define the g function
def g (n : ℕ) : ℚ :=
  (sigma n : ℚ) / n

-- Theorem statement
theorem g_difference : g 432 - g 216 = 5 / 54 := by sorry

end NUMINAMATH_CALUDE_g_difference_l1560_156016


namespace NUMINAMATH_CALUDE_right_triangle_with_constraint_l1560_156071

-- Define the triangle sides
def side1 (p q : ℝ) : ℝ := p
def side2 (p q : ℝ) : ℝ := p + q
def side3 (p q : ℝ) : ℝ := p + 2*q

-- Define the conditions
def is_right_triangle (p q : ℝ) : Prop :=
  (side3 p q)^2 = (side1 p q)^2 + (side2 p q)^2

def longest_side_constraint (p q : ℝ) : Prop :=
  side3 p q ≤ 12

-- Theorem statement
theorem right_triangle_with_constraint :
  ∃ (p q : ℝ),
    is_right_triangle p q ∧
    longest_side_constraint p q ∧
    p = (1 + Real.sqrt 7) / 2 ∧
    q = 1 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_with_constraint_l1560_156071


namespace NUMINAMATH_CALUDE_company_z_employees_l1560_156074

/-- The number of employees in Company Z having birthdays on Wednesday -/
def wednesday_birthdays : ℕ := 12

/-- The number of employees in Company Z having birthdays on any day other than Wednesday -/
def other_day_birthdays : ℕ := 11

/-- The number of days in a week -/
def days_in_week : ℕ := 7

theorem company_z_employees :
  let total_employees := wednesday_birthdays + (days_in_week - 1) * other_day_birthdays
  wednesday_birthdays > other_day_birthdays →
  total_employees = 78 := by
  sorry

end NUMINAMATH_CALUDE_company_z_employees_l1560_156074


namespace NUMINAMATH_CALUDE_normal_distribution_std_dev_l1560_156015

/-- Given a normal distribution with mean 12 and a value 9.6 that is 2 standard deviations
    below the mean, the standard deviation is 1.2 -/
theorem normal_distribution_std_dev (μ σ : ℝ) (h1 : μ = 12) (h2 : μ - 2 * σ = 9.6) :
  σ = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_normal_distribution_std_dev_l1560_156015


namespace NUMINAMATH_CALUDE_train_length_l1560_156028

/-- The length of a train given its speed and time to pass a fixed point -/
theorem train_length (speed : ℝ) (time : ℝ) (length : ℝ) : 
  speed = 50 → -- speed in km/h
  time = 5.399568034557236 → -- time in seconds
  length = speed * 1000 / 3600 * time → -- length in meters
  length = 75 := by sorry

end NUMINAMATH_CALUDE_train_length_l1560_156028


namespace NUMINAMATH_CALUDE_lees_friend_money_l1560_156097

/-- 
Given:
- Lee had $10
- The total cost of the meal was $15 (including tax)
- They received $3 in change
- The total amount paid was $18

Prove that Lee's friend had $8 initially.
-/
theorem lees_friend_money (lee_money : ℕ) (meal_cost : ℕ) (change : ℕ) (total_paid : ℕ)
  (h1 : lee_money = 10)
  (h2 : meal_cost = 15)
  (h3 : change = 3)
  (h4 : total_paid = 18)
  : total_paid - lee_money = 8 := by
  sorry

end NUMINAMATH_CALUDE_lees_friend_money_l1560_156097


namespace NUMINAMATH_CALUDE_expression_simplification_l1560_156020

theorem expression_simplification (a : ℝ) (h : a^2 + 2*a - 8 = 0) :
  ((a^2 - 4) / (a^2 - 4*a + 4) - a / (a - 2)) / ((a^2 + 2*a) / (a - 2)) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1560_156020


namespace NUMINAMATH_CALUDE_point_coordinates_l1560_156089

/-- A point in a plane rectangular coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- The fourth quadrant of a plane rectangular coordinate system -/
def fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The distance from a point to the x-axis -/
def distance_to_x_axis (p : Point) : ℝ :=
  |p.y|

/-- The distance from a point to the y-axis -/
def distance_to_y_axis (p : Point) : ℝ :=
  |p.x|

/-- The main theorem -/
theorem point_coordinates (p : Point) 
  (h1 : fourth_quadrant p) 
  (h2 : distance_to_x_axis p = 2) 
  (h3 : distance_to_y_axis p = 3) : 
  p = Point.mk 3 (-2) := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l1560_156089


namespace NUMINAMATH_CALUDE_tank_capacity_proof_l1560_156034

/-- Represents the filling rate of pipe A in liters per minute -/
def pipe_A_rate : ℕ := 40

/-- Represents the filling rate of pipe B in liters per minute -/
def pipe_B_rate : ℕ := 30

/-- Represents the draining rate of pipe C in liters per minute -/
def pipe_C_rate : ℕ := 20

/-- Represents the time in minutes it takes to fill the tank -/
def fill_time : ℕ := 48

/-- Represents the capacity of the tank in liters -/
def tank_capacity : ℕ := 780

/-- Theorem stating that given the pipe rates and fill time, the tank capacity is 780 liters -/
theorem tank_capacity_proof :
  pipe_A_rate = 40 →
  pipe_B_rate = 30 →
  pipe_C_rate = 20 →
  fill_time = 48 →
  tank_capacity = 780 :=
by sorry

end NUMINAMATH_CALUDE_tank_capacity_proof_l1560_156034


namespace NUMINAMATH_CALUDE_two_eyes_for_dog_l1560_156065

/-- Given a family that catches and distributes fish, calculate the number of fish eyes left for the dog. -/
def fish_eyes_for_dog (family_size : ℕ) (fish_per_person : ℕ) (eyes_per_fish : ℕ) (eyes_eaten : ℕ) : ℕ :=
  let total_fish := family_size * fish_per_person
  let total_eyes := total_fish * eyes_per_fish
  total_eyes - eyes_eaten

/-- Theorem stating that under the given conditions, 2 fish eyes remain for the dog. -/
theorem two_eyes_for_dog :
  fish_eyes_for_dog 3 4 2 22 = 2 :=
by sorry

end NUMINAMATH_CALUDE_two_eyes_for_dog_l1560_156065


namespace NUMINAMATH_CALUDE_consecutive_integers_cube_sum_l1560_156079

theorem consecutive_integers_cube_sum : 
  ∃ (a : ℕ), 
    (a > 0) ∧ 
    ((a - 1) * a * (a + 1) * (a + 2) = 12 * (4 * a + 2)) ∧ 
    ((a - 1)^3 + a^3 + (a + 1)^3 + (a + 2)^3 = 224) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_cube_sum_l1560_156079


namespace NUMINAMATH_CALUDE_sum_of_numbers_greater_than_threshold_l1560_156010

def numbers : List ℚ := [14/10, 9/10, 12/10, 5/10, 13/10]
def threshold : ℚ := 11/10

theorem sum_of_numbers_greater_than_threshold :
  (numbers.filter (λ x => x > threshold)).sum = 39/10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_greater_than_threshold_l1560_156010


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l1560_156025

theorem greatest_divisor_with_remainders : ∃ (n : ℕ), 
  (n ∣ (150 - 50)) ∧ 
  (n ∣ (230 - 5)) ∧ 
  (n ∣ (175 - 25)) ∧ 
  (∀ m : ℕ, m > n → (m ∣ (150 - 50)) → (m ∣ (230 - 5)) → ¬(m ∣ (175 - 25))) := by
  sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l1560_156025


namespace NUMINAMATH_CALUDE_problem_solution_l1560_156085

-- Define the function f
def f (a x : ℝ) : ℝ := |a - x|

-- Define the set A
def A : Set ℝ := {x | f (3/2) (2*x - 3/2) > 2 * f (3/2) (x + 2) + 2}

theorem problem_solution :
  (A = Set.Iio 0) ∧
  (∀ x₀ ∈ A, ∀ x : ℝ, f (3/2) (x₀ * x) ≥ x₀ * f (3/2) x + f (3/2) ((3/2) * x₀)) := by
  sorry


end NUMINAMATH_CALUDE_problem_solution_l1560_156085


namespace NUMINAMATH_CALUDE_share_price_increase_l1560_156068

theorem share_price_increase (P : ℝ) (h : P > 0) : 
  let first_quarter_price := P * 1.30
  let second_quarter_increase := 0.15384615384615374
  let second_quarter_price := first_quarter_price * (1 + second_quarter_increase)
  (second_quarter_price - P) / P = 0.50 := by sorry

end NUMINAMATH_CALUDE_share_price_increase_l1560_156068


namespace NUMINAMATH_CALUDE_binary_octal_conversion_l1560_156013

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.foldr (fun b acc => 2 * acc + if b then 1 else 0) 0

/-- Converts a decimal number to its octal representation -/
def decimal_to_octal (n : ℕ) : List ℕ :=
  if n < 8 then [n]
  else (n % 8) :: decimal_to_octal (n / 8)

/-- The binary representation of the number in question -/
def binary_number : List Bool :=
  [true, false, true, true, false, true, true, true, false]

theorem binary_octal_conversion :
  (binary_to_decimal binary_number = 54) ∧
  (decimal_to_octal 54 = [6, 6]) :=
by sorry

end NUMINAMATH_CALUDE_binary_octal_conversion_l1560_156013


namespace NUMINAMATH_CALUDE_quadratic_function_value_at_three_l1560_156063

/-- A quadratic function f(x) = ax^2 + bx + c with roots at x=1 and x=5, and minimum value 36 -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_function_value_at_three
  (a b c : ℝ)
  (root_one : QuadraticFunction a b c 1 = 0)
  (root_five : QuadraticFunction a b c 5 = 0)
  (min_value : ∀ x, QuadraticFunction a b c x ≥ 36)
  (attains_min : ∃ x, QuadraticFunction a b c x = 36) :
  QuadraticFunction a b c 3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_value_at_three_l1560_156063


namespace NUMINAMATH_CALUDE_frank_planted_two_seeds_per_orange_l1560_156075

/-- The number of oranges Betty picked -/
def betty_oranges : ℕ := 15

/-- The number of oranges Bill picked -/
def bill_oranges : ℕ := 12

/-- The number of oranges Frank picked -/
def frank_oranges : ℕ := 3 * (betty_oranges + bill_oranges)

/-- The number of oranges each tree contains -/
def oranges_per_tree : ℕ := 5

/-- The total number of oranges Philip can pick -/
def philip_total_oranges : ℕ := 810

/-- The number of seeds Frank planted from each of his oranges -/
def seeds_per_orange : ℕ := philip_total_oranges / oranges_per_tree / frank_oranges

theorem frank_planted_two_seeds_per_orange : seeds_per_orange = 2 := by
  sorry

end NUMINAMATH_CALUDE_frank_planted_two_seeds_per_orange_l1560_156075


namespace NUMINAMATH_CALUDE_tennis_players_count_l1560_156056

theorem tennis_players_count (total : ℕ) (badminton : ℕ) (neither : ℕ) (both : ℕ) 
  (h1 : total = 35)
  (h2 : badminton = 15)
  (h3 : neither = 5)
  (h4 : both = 3) :
  ∃ tennis : ℕ, tennis = 18 ∧ 
  tennis = total - neither - (badminton - both) := by
  sorry

end NUMINAMATH_CALUDE_tennis_players_count_l1560_156056


namespace NUMINAMATH_CALUDE_inequality_solution_l1560_156066

theorem inequality_solution :
  ∀ x : ℕ, 1 + x ≥ 2 * x - 1 ↔ x ∈ ({0, 1, 2} : Set ℕ) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l1560_156066


namespace NUMINAMATH_CALUDE_integral_exp_plus_2x_equals_e_l1560_156072

theorem integral_exp_plus_2x_equals_e :
  ∫ x in (0:ℝ)..1, (Real.exp x + 2 * x) = Real.exp 1 - 1 := by
  sorry

end NUMINAMATH_CALUDE_integral_exp_plus_2x_equals_e_l1560_156072


namespace NUMINAMATH_CALUDE_quadratic_root_implies_a_bound_l1560_156024

theorem quadratic_root_implies_a_bound (a : ℝ) (h1 : a > 0) 
  (h2 : 3^2 - 5/3 * a * 3 - a^2 = 0) : 1 < a ∧ a < 3/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_a_bound_l1560_156024


namespace NUMINAMATH_CALUDE_jack_and_jill_speed_jack_and_jill_speed_proof_l1560_156023

/-- The common speed of Jack and Jill given their walking conditions -/
theorem jack_and_jill_speed : ℝ → Prop :=
  fun (x : ℝ) ↦ 
    let jack_speed := x^2 - 11*x - 22
    let jill_distance := x^2 - 3*x - 54
    let jill_time := x + 6
    let jill_speed := jill_distance / jill_time
    (jack_speed = jill_speed) → (jack_speed = 4)

/-- Proof of the theorem -/
theorem jack_and_jill_speed_proof : ∃ x : ℝ, jack_and_jill_speed x :=
  sorry

end NUMINAMATH_CALUDE_jack_and_jill_speed_jack_and_jill_speed_proof_l1560_156023


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l1560_156005

theorem other_root_of_quadratic (m : ℝ) : 
  (∃ x : ℝ, 7 * x^2 + m * x - 6 = 0 ∧ x = -3) →
  (7 * (2/7)^2 + m * (2/7) - 6 = 0) :=
by sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l1560_156005


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l1560_156009

theorem quadratic_equations_solutions :
  (∃ x : ℝ, x^2 - 6*x + 4 = 0) ∧
  (∃ x : ℝ, (3*x - 1)^2 - 4*x^2 = 0) ∧
  (∀ x : ℝ, x^2 - 6*x + 4 = 0 ↔ x = 3 + Real.sqrt 5 ∨ x = 3 - Real.sqrt 5) ∧
  (∀ x : ℝ, (3*x - 1)^2 - 4*x^2 = 0 ↔ x = 1/5 ∨ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l1560_156009


namespace NUMINAMATH_CALUDE_angle_Y_measure_l1560_156064

-- Define the structure for lines and angles
structure Geometry where
  Line : Type
  Angle : Type
  measure : Angle → ℝ
  parallel : Line → Line → Prop
  intersect : Line → Line → Prop
  angleOn : Line → Angle → Prop
  transversal : Line → Line → Line → Prop

-- State the theorem
theorem angle_Y_measure (G : Geometry) 
  (p q t yz : G.Line) (X Z Y : G.Angle) :
  G.parallel p q →
  G.parallel p yz →
  G.parallel q yz →
  G.transversal t p q →
  G.intersect t yz →
  G.angleOn p X →
  G.angleOn q Z →
  G.measure X = 100 →
  G.measure Z = 110 →
  G.measure Y = 40 := by
  sorry


end NUMINAMATH_CALUDE_angle_Y_measure_l1560_156064


namespace NUMINAMATH_CALUDE_solution_set_f_geq_1_max_value_f_minus_x_squared_plus_x_l1560_156003

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| - |x - 2|

-- Theorem 1: The solution set of f(x) ≥ 1 is {x | x ≥ 1}
theorem solution_set_f_geq_1 :
  {x : ℝ | f x ≥ 1} = {x : ℝ | x ≥ 1} := by sorry

-- Theorem 2: The maximum value of f(x) - x^2 + x is 5/4
theorem max_value_f_minus_x_squared_plus_x :
  ∃ (x : ℝ), ∀ (y : ℝ), f y - y^2 + y ≤ f x - x^2 + x ∧ f x - x^2 + x = 5/4 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_geq_1_max_value_f_minus_x_squared_plus_x_l1560_156003


namespace NUMINAMATH_CALUDE_pentagon_smallest_angle_l1560_156043

theorem pentagon_smallest_angle (a b c d e : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 →
  a + b + c + d + e = 540 →
  b = 4/3 * a →
  c = 5/3 * a →
  d = 2 * a →
  e = 7/3 * a →
  a = 64.8 :=
by sorry

end NUMINAMATH_CALUDE_pentagon_smallest_angle_l1560_156043


namespace NUMINAMATH_CALUDE_simplify_expression_l1560_156060

theorem simplify_expression (x y : ℝ) : (3 * x^2 * y^3)^4 = 81 * x^8 * y^12 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1560_156060


namespace NUMINAMATH_CALUDE_can_display_rows_l1560_156035

/-- Represents a display of cans arranged in rows. -/
structure CanDisplay where
  firstRowCans : ℕ  -- Number of cans in the first row
  rowIncrement : ℕ  -- Increment in number of cans for each subsequent row
  totalCans : ℕ     -- Total number of cans in the display

/-- Calculates the number of rows in a can display. -/
def numberOfRows (display : CanDisplay) : ℕ :=
  sorry

/-- Theorem stating that a display with 2 cans in the first row,
    incrementing by 3 cans each row, and totaling 120 cans has 9 rows. -/
theorem can_display_rows :
  let display : CanDisplay := {
    firstRowCans := 2,
    rowIncrement := 3,
    totalCans := 120
  }
  numberOfRows display = 9 := by sorry

end NUMINAMATH_CALUDE_can_display_rows_l1560_156035


namespace NUMINAMATH_CALUDE_sum_of_last_two_digits_of_series_l1560_156044

def modified_fibonacci_factorial_series : List Nat :=
  [1, 2, 3, 4, 7, 11, 18, 29, 47, 76]

def last_two_digits (n : Nat) : Nat :=
  n % 100

def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem sum_of_last_two_digits_of_series :
  (modified_fibonacci_factorial_series.map (λ n => last_two_digits (factorial n))).sum = 73 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_last_two_digits_of_series_l1560_156044


namespace NUMINAMATH_CALUDE_nadines_pebbles_l1560_156036

/-- The number of pebbles Nadine has -/
def total_pebbles (white red blue green : ℕ) : ℕ := white + red + blue + green

/-- Theorem stating the total number of pebbles Nadine has -/
theorem nadines_pebbles :
  ∀ (white red blue green : ℕ),
  white = 20 →
  red = white / 2 →
  blue = red / 3 →
  green = blue + 5 →
  total_pebbles white red blue green = 41 := by
sorry

#eval total_pebbles 20 10 3 8

end NUMINAMATH_CALUDE_nadines_pebbles_l1560_156036


namespace NUMINAMATH_CALUDE_time_to_weave_cloth_l1560_156039

/-- Represents the industrial loom's weaving rate and characteristics -/
structure Loom where
  rate : Real
  sample_time : Real
  sample_cloth : Real

/-- Theorem: Time to weave cloth -/
theorem time_to_weave_cloth (loom : Loom) (x : Real) :
  loom.rate = 0.128 ∧ 
  loom.sample_time = 195.3125 ∧ 
  loom.sample_cloth = 25 →
  x / loom.rate = x / 0.128 := by
  sorry

#check time_to_weave_cloth

end NUMINAMATH_CALUDE_time_to_weave_cloth_l1560_156039


namespace NUMINAMATH_CALUDE_friend_name_probability_l1560_156059

def total_cards : ℕ := 15
def cybil_cards : ℕ := 6
def ronda_cards : ℕ := 9
def cards_drawn : ℕ := 3

theorem friend_name_probability : 
  (1 : ℚ) - (Nat.choose ronda_cards cards_drawn : ℚ) / (Nat.choose total_cards cards_drawn)
         - (Nat.choose cybil_cards cards_drawn : ℚ) / (Nat.choose total_cards cards_drawn)
  = 351 / 455 := by sorry

end NUMINAMATH_CALUDE_friend_name_probability_l1560_156059


namespace NUMINAMATH_CALUDE_exists_close_points_on_graphs_l1560_156027

open Real

/-- The function f(x) = x^4 -/
def f (x : ℝ) : ℝ := x^4

/-- The function g(x) = x^4 + x^2 + x + 1 -/
def g (x : ℝ) : ℝ := x^4 + x^2 + x + 1

/-- Theorem stating the existence of points A and B on the graphs of f and g with distance < 1/100 -/
theorem exists_close_points_on_graphs :
  ∃ (u v : ℝ), |u - v| < 1/100 ∧ f v = g u := by sorry

end NUMINAMATH_CALUDE_exists_close_points_on_graphs_l1560_156027
