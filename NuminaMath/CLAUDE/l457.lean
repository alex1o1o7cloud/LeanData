import Mathlib

namespace NUMINAMATH_CALUDE_sprinkles_remaining_l457_45708

theorem sprinkles_remaining (initial_cans : ℕ) (remaining_cans : ℕ) : 
  initial_cans = 12 →
  remaining_cans = initial_cans / 2 - 3 →
  remaining_cans = 3 := by
sorry

end NUMINAMATH_CALUDE_sprinkles_remaining_l457_45708


namespace NUMINAMATH_CALUDE_erdos_szekeres_l457_45717

theorem erdos_szekeres (m n : ℕ) (seq : Fin (m * n + 1) → ℝ) :
  (∃ (subseq : Fin (m + 1) → Fin (m * n + 1)),
    (∀ i j, i < j → seq (subseq i) ≤ seq (subseq j))) ∨
  (∃ (subseq : Fin (n + 1) → Fin (m * n + 1)),
    (∀ i j, i < j → seq (subseq i) ≥ seq (subseq j))) :=
sorry

end NUMINAMATH_CALUDE_erdos_szekeres_l457_45717


namespace NUMINAMATH_CALUDE_cell_phone_price_l457_45729

/-- The price of a cell phone given the total cost and monthly payments --/
theorem cell_phone_price (total_cost : ℕ) (monthly_payment : ℕ) (num_months : ℕ) 
  (h1 : total_cost = 30)
  (h2 : monthly_payment = 7)
  (h3 : num_months = 4) :
  total_cost - (monthly_payment * num_months) = 2 := by
  sorry

end NUMINAMATH_CALUDE_cell_phone_price_l457_45729


namespace NUMINAMATH_CALUDE_election_total_votes_l457_45746

-- Define the set of candidates
inductive Candidate : Type
  | Alicia : Candidate
  | Brenda : Candidate
  | Colby : Candidate
  | David : Candidate

-- Define the election
structure Election where
  totalVotes : ℕ
  brendaVotes : ℕ
  brendaPercentage : ℚ

-- Theorem statement
theorem election_total_votes (e : Election) 
  (h1 : e.brendaVotes = 40)
  (h2 : e.brendaPercentage = 1/4) :
  e.totalVotes = 160 := by
  sorry


end NUMINAMATH_CALUDE_election_total_votes_l457_45746


namespace NUMINAMATH_CALUDE_range_of_m_l457_45711

theorem range_of_m (m : ℝ) : 
  (¬ ∃ x : ℝ, x ∈ Set.Icc (-1) m ∧ m > -1 ∧ |x| - 1 > 0) → 
  m ∈ Set.Ioo (-1) 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l457_45711


namespace NUMINAMATH_CALUDE_smallest_x_for_perfect_cube_l457_45741

def certain_number : ℕ := 1152

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m^3

theorem smallest_x_for_perfect_cube :
  ∃! x : ℕ, x > 0 ∧ is_perfect_cube (certain_number * x) ∧
    ∀ y : ℕ, y > 0 ∧ y < x → ¬is_perfect_cube (certain_number * y) ∧
    certain_number * x = 12 * certain_number :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_x_for_perfect_cube_l457_45741


namespace NUMINAMATH_CALUDE_floor_squared_sum_four_l457_45737

theorem floor_squared_sum_four (x y : ℝ) : 
  (Int.floor x)^2 + (Int.floor y)^2 = 4 ↔ 
    ((2 ≤ x ∧ x < 3 ∧ 0 ≤ y ∧ y < 1) ∨
     (0 ≤ x ∧ x < 1 ∧ 2 ≤ y ∧ y < 3) ∨
     (-2 ≤ x ∧ x < -1 ∧ 0 ≤ y ∧ y < 1) ∨
     (0 ≤ x ∧ x < 1 ∧ -2 ≤ y ∧ y < -1)) :=
by sorry

end NUMINAMATH_CALUDE_floor_squared_sum_four_l457_45737


namespace NUMINAMATH_CALUDE_expression_simplification_l457_45762

theorem expression_simplification : 
  (3 * 5 * 7) / (9 * 11 * 13) * (7 * 9 * 11 * 15) / (3 * 5 * 14) = 15 / 26 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l457_45762


namespace NUMINAMATH_CALUDE_vision_assistance_l457_45787

theorem vision_assistance (total : ℕ) (glasses_percent : ℚ) (contacts_percent : ℚ)
  (h_total : total = 40)
  (h_glasses : glasses_percent = 25 / 100)
  (h_contacts : contacts_percent = 40 / 100) :
  total - (total * glasses_percent).floor - (total * contacts_percent).floor = 14 := by
  sorry

end NUMINAMATH_CALUDE_vision_assistance_l457_45787


namespace NUMINAMATH_CALUDE_sin_two_alpha_zero_l457_45732

open Real

theorem sin_two_alpha_zero (α : ℝ) (f : ℝ → ℝ) (h : f = λ x => sin x - cos x) (h1 : f α = 1) : sin (2 * α) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_two_alpha_zero_l457_45732


namespace NUMINAMATH_CALUDE_union_of_a_and_b_l457_45747

def U : Set Nat := {0, 1, 2, 3, 4}

theorem union_of_a_and_b (A B : Set Nat) 
  (h1 : U = {0, 1, 2, 3, 4})
  (h2 : (U \ A) = {1, 2})
  (h3 : B = {1, 3}) :
  A ∪ B = {0, 1, 3, 4} := by
sorry

end NUMINAMATH_CALUDE_union_of_a_and_b_l457_45747


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l457_45740

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  x₁ = 5 ∧ x₂ = -1 ∧ 
  x₁^2 - 4*x₁ - 5 = 0 ∧ 
  x₂^2 - 4*x₂ - 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l457_45740


namespace NUMINAMATH_CALUDE_total_noodles_and_pirates_l457_45738

theorem total_noodles_and_pirates (pirates : ℕ) (noodle_difference : ℕ) : 
  pirates = 45 → noodle_difference = 7 → pirates + (pirates - noodle_difference) = 83 := by
  sorry

end NUMINAMATH_CALUDE_total_noodles_and_pirates_l457_45738


namespace NUMINAMATH_CALUDE_phone_not_answered_probability_l457_45710

theorem phone_not_answered_probability 
  (p1 : ℝ) (p2 : ℝ) (p3 : ℝ) (p4 : ℝ)
  (h1 : p1 = 0.1) (h2 : p2 = 0.3) (h3 : p3 = 0.4) (h4 : p4 = 0.1) :
  (1 - p1) * (1 - p2) * (1 - p3) * (1 - p4) = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_phone_not_answered_probability_l457_45710


namespace NUMINAMATH_CALUDE_min_overlap_mozart_bach_l457_45767

theorem min_overlap_mozart_bach (total : ℕ) (mozart : ℕ) (bach : ℕ) 
  (h_total : total = 200)
  (h_mozart : mozart = 160)
  (h_bach : bach = 145)
  : mozart + bach - total ≥ 105 := by
  sorry

end NUMINAMATH_CALUDE_min_overlap_mozart_bach_l457_45767


namespace NUMINAMATH_CALUDE_set_equality_l457_45766

theorem set_equality : {x : ℕ | x - 3 < 2} = {0, 1, 2, 3, 4} := by
  sorry

end NUMINAMATH_CALUDE_set_equality_l457_45766


namespace NUMINAMATH_CALUDE_cos_phase_shift_l457_45792

/-- The phase shift of y = cos(2x + π/2) is -π/4 --/
theorem cos_phase_shift : 
  let f := fun x => Real.cos (2 * x + π / 2)
  let phase_shift := fun (B C : ℝ) => -C / B
  phase_shift 2 (π / 2) = -π / 4 := by
sorry

end NUMINAMATH_CALUDE_cos_phase_shift_l457_45792


namespace NUMINAMATH_CALUDE_simplify_A_minus_B_A_minus_B_value_l457_45749

/-- Given two real numbers a and b, we define A and B as follows -/
def A (a b : ℝ) : ℝ := (a + b)^2 - 3 * b^2

def B (a b : ℝ) : ℝ := 2 * (a + b) * (a - b) - 3 * a * b

/-- Theorem stating that A - B simplifies to -a^2 + 5ab -/
theorem simplify_A_minus_B (a b : ℝ) : A a b - B a b = -a^2 + 5*a*b := by sorry

/-- Theorem stating that if (a-3)^2 + |b-4| = 0, then A - B = 51 -/
theorem A_minus_B_value (a b : ℝ) (h : (a - 3)^2 + |b - 4| = 0) : A a b - B a b = 51 := by sorry

end NUMINAMATH_CALUDE_simplify_A_minus_B_A_minus_B_value_l457_45749


namespace NUMINAMATH_CALUDE_sum_of_digits_of_7_pow_25_l457_45795

/-- The sum of the tens digit and the ones digit of 7^25 -/
def sum_of_digits : ℕ :=
  let n : ℕ := 7^25
  (n / 10 % 10) + (n % 10)

/-- Theorem stating that the sum of the tens digit and the ones digit of 7^25 is 7 -/
theorem sum_of_digits_of_7_pow_25 : sum_of_digits = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_7_pow_25_l457_45795


namespace NUMINAMATH_CALUDE_cylinder_volume_constant_l457_45726

/-- Given a cube with side length 3 and a cylinder with the same surface area,
    if the volume of the cylinder is (M * sqrt(6)) / sqrt(π),
    then M = 9 * sqrt(6) * π -/
theorem cylinder_volume_constant (M : ℝ) : 
  let cube_side : ℝ := 3
  let cube_surface_area : ℝ := 6 * cube_side^2
  ∃ (r h : ℝ),
    (2 * π * r^2 + 2 * π * r * h = cube_surface_area) ∧ 
    (π * r^2 * h = (M * Real.sqrt 6) / Real.sqrt π) →
    M = 9 * Real.sqrt 6 * π :=
by sorry

end NUMINAMATH_CALUDE_cylinder_volume_constant_l457_45726


namespace NUMINAMATH_CALUDE_meeting_point_distance_l457_45770

/-- Proves that the distance between Jack and Jill's meeting point and the hilltop is 35/27 km -/
theorem meeting_point_distance (total_distance : ℝ) (uphill_distance : ℝ)
  (jack_start_earlier : ℝ) (jack_uphill_speed : ℝ) (jack_downhill_speed : ℝ)
  (jill_uphill_speed : ℝ) :
  total_distance = 10 →
  uphill_distance = 5 →
  jack_start_earlier = 1/6 →
  jack_uphill_speed = 15 →
  jack_downhill_speed = 20 →
  jill_uphill_speed = 16 →
  ∃ (meeting_point_distance : ℝ), meeting_point_distance = 35/27 := by
  sorry

end NUMINAMATH_CALUDE_meeting_point_distance_l457_45770


namespace NUMINAMATH_CALUDE_unique_number_l457_45759

def is_valid_number (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000 ∧ 
  n / 100000 = 1 ∧
  (n % 100000) * 10 + 1 = 3 * n

theorem unique_number : ∃! n : ℕ, is_valid_number n :=
  sorry

end NUMINAMATH_CALUDE_unique_number_l457_45759


namespace NUMINAMATH_CALUDE_john_coffee_consumption_l457_45702

/-- Represents the number of fluid ounces in a gallon -/
def gallonToOunces : ℚ := 128

/-- Represents the number of fluid ounces in a standard cup -/
def cupToOunces : ℚ := 8

/-- Represents the number of days between John's coffee purchases -/
def purchaseInterval : ℚ := 4

/-- Represents the fraction of a gallon John buys each time -/
def purchaseAmount : ℚ := 1/2

/-- Theorem stating that John drinks 2 cups of coffee per day -/
theorem john_coffee_consumption :
  let cupsPerPurchase := purchaseAmount * gallonToOunces / cupToOunces
  cupsPerPurchase / purchaseInterval = 2 := by sorry

end NUMINAMATH_CALUDE_john_coffee_consumption_l457_45702


namespace NUMINAMATH_CALUDE_equation_solution_l457_45731

theorem equation_solution (x : ℚ) (h : x ≠ -2) :
  (4 * x / (x + 2) - 2 / (x + 2) = 3 / (x + 2)) → x = 5 / 4 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l457_45731


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_bounds_l457_45757

/-- Given a triangle with sides a ≤ b ≤ c and corresponding altitudes ma ≥ mb ≥ mc,
    the radius ρ of the inscribed circle satisfies mc/3 ≤ ρ ≤ ma/3 -/
theorem inscribed_circle_radius_bounds (a b c ma mb mc ρ : ℝ) 
  (h_sides : a ≤ b ∧ b ≤ c)
  (h_altitudes : ma ≥ mb ∧ mb ≥ mc)
  (h_inradius : ρ > 0)
  (h_area : ρ * (a + b + c) = a * ma)
  (h_area_alt : ρ * (a + b + c) = c * mc) :
  mc / 3 ≤ ρ ∧ ρ ≤ ma / 3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_bounds_l457_45757


namespace NUMINAMATH_CALUDE_circle_area_tripled_l457_45713

theorem circle_area_tripled (r n : ℝ) : 
  (π * (r + n)^2 = 3 * π * r^2) → (r = n * (Real.sqrt 3 - 1) / 2) :=
by sorry

end NUMINAMATH_CALUDE_circle_area_tripled_l457_45713


namespace NUMINAMATH_CALUDE_problem_statement_l457_45724

theorem problem_statement : (1 / (64^(1/3))^9) * 8^6 = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l457_45724


namespace NUMINAMATH_CALUDE_y_gets_20_percent_more_than_z_l457_45706

/-- The problem setup with given conditions -/
def problem_setup (x y z : ℝ) : Prop :=
  x = y * 1.25 ∧  -- x gets 25% more than y
  740 = x + y + z ∧  -- total amount is 740
  z = 200  -- z's share is 200

/-- The theorem to prove -/
theorem y_gets_20_percent_more_than_z 
  (x y z : ℝ) (h : problem_setup x y z) : y = z * 1.2 := by
  sorry


end NUMINAMATH_CALUDE_y_gets_20_percent_more_than_z_l457_45706


namespace NUMINAMATH_CALUDE_luisa_pet_store_distance_l457_45750

theorem luisa_pet_store_distance (grocery_store_distance : ℝ) (mall_distance : ℝ) (home_distance : ℝ) 
  (miles_per_gallon : ℝ) (cost_per_gallon : ℝ) (total_cost : ℝ) :
  grocery_store_distance = 10 →
  mall_distance = 6 →
  home_distance = 9 →
  miles_per_gallon = 15 →
  cost_per_gallon = 3.5 →
  total_cost = 7 →
  ∃ (pet_store_distance : ℝ),
    pet_store_distance = 5 ∧
    grocery_store_distance + mall_distance + pet_store_distance + home_distance = 
      (total_cost / cost_per_gallon) * miles_per_gallon :=
by sorry

end NUMINAMATH_CALUDE_luisa_pet_store_distance_l457_45750


namespace NUMINAMATH_CALUDE_supplement_of_complementary_l457_45765

/-- Two angles are complementary if their sum is 90 degrees -/
def complementary (α β : ℝ) : Prop := α + β = 90

/-- The supplement of an angle is 180 degrees minus the angle -/
def supplement (θ : ℝ) : ℝ := 180 - θ

/-- 
If two angles α and β are complementary, 
then the supplement of α is 90 degrees greater than β 
-/
theorem supplement_of_complementary (α β : ℝ) 
  (h : complementary α β) : 
  supplement α = β + 90 := by sorry

end NUMINAMATH_CALUDE_supplement_of_complementary_l457_45765


namespace NUMINAMATH_CALUDE_tangent_line_equations_l457_45790

/-- The equations of the lines passing through point (1,1) and tangent to the curve y = x³ + 1 -/
theorem tangent_line_equations : 
  ∃ (m b : ℝ), 
    (∀ x y : ℝ, y = m * x + b → (x = 1 ∧ y = 1)) ∧ 
    (∃ x₀ : ℝ, 
      (x₀^3 + 1 = m * x₀ + b) ∧ 
      (3 * x₀^2 = m)) ∧
    ((m = 0 ∧ b = 1) ∨ (m = 27/4 ∧ b = -23/4)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equations_l457_45790


namespace NUMINAMATH_CALUDE_only_baseball_count_l457_45754

/-- Represents the number of people in different categories in a class --/
structure ClassSports where
  total : ℕ
  both : ℕ
  onlyFootball : ℕ
  neither : ℕ

/-- Theorem stating the number of people who only like baseball --/
theorem only_baseball_count (c : ClassSports) 
  (h1 : c.total = 16)
  (h2 : c.both = 5)
  (h3 : c.onlyFootball = 3)
  (h4 : c.neither = 6) :
  c.total - (c.both + c.onlyFootball + c.neither) = 2 :=
sorry

end NUMINAMATH_CALUDE_only_baseball_count_l457_45754


namespace NUMINAMATH_CALUDE_forty_percent_of_number_l457_45786

theorem forty_percent_of_number (N : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 15 → (40/100 : ℝ) * N = 180 := by
  sorry

end NUMINAMATH_CALUDE_forty_percent_of_number_l457_45786


namespace NUMINAMATH_CALUDE_olaf_total_cars_l457_45733

/-- The number of toy cars in Olaf's collection --/
def total_cars (initial : ℕ) (grandpa uncle dad mum auntie : ℕ) : ℕ :=
  initial + grandpa + uncle + dad + mum + auntie

/-- The conditions of Olaf's toy car collection problem --/
def olaf_problem (initial grandpa uncle dad mum auntie : ℕ) : Prop :=
  initial = 150 ∧
  grandpa = 2 * uncle ∧
  dad = 10 ∧
  mum = dad + 5 ∧
  auntie = uncle + 1 ∧
  auntie = 6

/-- Theorem stating that Olaf's total number of cars is 196 --/
theorem olaf_total_cars :
  ∀ initial grandpa uncle dad mum auntie : ℕ,
  olaf_problem initial grandpa uncle dad mum auntie →
  total_cars initial grandpa uncle dad mum auntie = 196 :=
by
  sorry


end NUMINAMATH_CALUDE_olaf_total_cars_l457_45733


namespace NUMINAMATH_CALUDE_johns_per_sheet_price_l457_45709

def johns_sitting_fee : ℝ := 125
def sams_sitting_fee : ℝ := 140
def sams_per_sheet : ℝ := 1.50
def num_sheets : ℝ := 12

theorem johns_per_sheet_price (johns_per_sheet : ℝ) : 
  johns_per_sheet * num_sheets + johns_sitting_fee = 
  sams_per_sheet * num_sheets + sams_sitting_fee → 
  johns_per_sheet = 2.75 := by
sorry

end NUMINAMATH_CALUDE_johns_per_sheet_price_l457_45709


namespace NUMINAMATH_CALUDE_real_number_line_bijection_l457_45735

-- Define the number line as a type
def NumberLine : Type := ℝ

-- Define the bijection between real numbers and points on the number line
def realToPoint : ℝ → NumberLine := id

-- Statement: There is a one-to-one correspondence between real numbers and points on the number line
theorem real_number_line_bijection : Function.Bijective realToPoint := by
  sorry

end NUMINAMATH_CALUDE_real_number_line_bijection_l457_45735


namespace NUMINAMATH_CALUDE_furniture_dealer_profit_l457_45705

/-- Calculates the gross profit for a furniture dealer selling a desk -/
theorem furniture_dealer_profit
  (purchase_price : ℝ)
  (markup_percentage : ℝ)
  (discount_percentage : ℝ)
  (h1 : purchase_price = 150)
  (h2 : markup_percentage = 0.5)
  (h3 : discount_percentage = 0.2) :
  let selling_price := purchase_price / (1 - markup_percentage)
  let discounted_price := selling_price * (1 - discount_percentage)
  let gross_profit := discounted_price - purchase_price
  gross_profit = 90 := by sorry

end NUMINAMATH_CALUDE_furniture_dealer_profit_l457_45705


namespace NUMINAMATH_CALUDE_smallest_integer_problem_l457_45773

theorem smallest_integer_problem (a b c : ℕ+) : 
  (a : ℝ) + b + c = 90 ∧ 
  2 * a = 3 * b ∧ 
  2 * a = 5 * c ∧ 
  (a : ℝ) * b * c < 22000 → 
  a = 18 := by
sorry

end NUMINAMATH_CALUDE_smallest_integer_problem_l457_45773


namespace NUMINAMATH_CALUDE_house_construction_fraction_l457_45744

theorem house_construction_fraction (total : ℕ) (additional : ℕ) (remaining : ℕ) 
  (h_total : total = 2000)
  (h_additional : additional = 300)
  (h_remaining : remaining = 500) :
  (total - additional - remaining : ℚ) / total = 3 / 5 :=
sorry

end NUMINAMATH_CALUDE_house_construction_fraction_l457_45744


namespace NUMINAMATH_CALUDE_g_fixed_points_l457_45774

def g (x : ℝ) : ℝ := x^2 - 5*x

theorem g_fixed_points (x : ℝ) : g (g x) = g x ↔ x = -1 ∨ x = 0 ∨ x = 5 ∨ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_g_fixed_points_l457_45774


namespace NUMINAMATH_CALUDE_bill_with_late_charges_l457_45756

/-- The final bill amount after two late charges -/
def final_bill_amount (original_bill : ℝ) (first_charge_rate : ℝ) (second_charge_rate : ℝ) : ℝ :=
  original_bill * (1 + first_charge_rate) * (1 + second_charge_rate)

/-- Theorem stating the final bill amount after specific late charges -/
theorem bill_with_late_charges :
  final_bill_amount 500 0.02 0.03 = 525.30 := by
  sorry

end NUMINAMATH_CALUDE_bill_with_late_charges_l457_45756


namespace NUMINAMATH_CALUDE_spirit_mixture_problem_l457_45748

/-- Given three vessels a, b, and c with spirit concentrations of 45%, 30%, and 10% respectively,
    and a mixture of x litres from vessel a, 5 litres from vessel b, and 6 litres from vessel c
    resulting in a 26% spirit concentration, prove that x = 4 litres. -/
theorem spirit_mixture_problem (x : ℝ) :
  (0.45 * x + 0.30 * 5 + 0.10 * 6) / (x + 5 + 6) = 0.26 → x = 4 := by
  sorry

#check spirit_mixture_problem

end NUMINAMATH_CALUDE_spirit_mixture_problem_l457_45748


namespace NUMINAMATH_CALUDE_credit_card_balance_calculation_l457_45720

/-- Calculates the final balance on a credit card after two interest applications -/
def final_balance (initial_balance : ℝ) (interest_rate : ℝ) (additional_charge : ℝ) : ℝ :=
  let balance_after_first_interest := initial_balance * (1 + interest_rate)
  let balance_before_second_interest := balance_after_first_interest + additional_charge
  balance_before_second_interest * (1 + interest_rate)

/-- Theorem stating that given the specific conditions, the final balance is $96.00 -/
theorem credit_card_balance_calculation :
  final_balance 50 0.2 20 = 96 := by
  sorry

#eval final_balance 50 0.2 20

end NUMINAMATH_CALUDE_credit_card_balance_calculation_l457_45720


namespace NUMINAMATH_CALUDE_tank_fill_time_l457_45707

/-- The time it takes to fill a tank with two pipes and a leak -/
theorem tank_fill_time (pipe1_time pipe2_time : ℝ) (leak_fraction : ℝ) : 
  pipe1_time = 20 →
  pipe2_time = 30 →
  leak_fraction = 1/3 →
  (1 / ((1 / pipe1_time + 1 / pipe2_time) * (1 - leak_fraction))) = 18 := by
  sorry

end NUMINAMATH_CALUDE_tank_fill_time_l457_45707


namespace NUMINAMATH_CALUDE_solution_set_for_a_zero_range_of_a_for_solution_exists_l457_45799

-- Define the functions f and g
def f (x : ℝ) : ℝ := abs (x + 1)
def g (a : ℝ) (x : ℝ) : ℝ := 2 * abs x + a

-- Theorem for part (I)
theorem solution_set_for_a_zero :
  {x : ℝ | f x ≥ g 0 x} = Set.Icc (-1/3) 1 := by sorry

-- Theorem for part (II)
theorem range_of_a_for_solution_exists :
  {a : ℝ | ∃ x, f x ≥ g a x} = Set.Iic 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_for_a_zero_range_of_a_for_solution_exists_l457_45799


namespace NUMINAMATH_CALUDE_not_p_necessary_not_sufficient_for_not_q_implies_m_ge_11_div_2_l457_45758

/-- Proposition p: -x^2 + 8x + 20 ≥ 0 -/
def p (x : ℝ) : Prop := -x^2 + 8*x + 20 ≥ 0

/-- Proposition q: x^2 + 2x + 1 - 4m^2 ≤ 0 -/
def q (x m : ℝ) : Prop := x^2 + 2*x + 1 - 4*m^2 ≤ 0

/-- If ¬p is a necessary but not sufficient condition for ¬q when m > 0, then m ≥ 11/2 -/
theorem not_p_necessary_not_sufficient_for_not_q_implies_m_ge_11_div_2 :
  ∀ m : ℝ, m > 0 →
  (∀ x : ℝ, (¬q x m → ¬p x) ∧ (∃ x : ℝ, ¬p x ∧ q x m)) →
  m ≥ 11/2 :=
sorry

end NUMINAMATH_CALUDE_not_p_necessary_not_sufficient_for_not_q_implies_m_ge_11_div_2_l457_45758


namespace NUMINAMATH_CALUDE_current_age_proof_l457_45718

theorem current_age_proof (my_age : ℕ) (son_age : ℕ) : 
  (my_age - 9 = 5 * (son_age - 9)) →
  (my_age = 3 * son_age) →
  my_age = 54 := by
  sorry

end NUMINAMATH_CALUDE_current_age_proof_l457_45718


namespace NUMINAMATH_CALUDE_point_on_curve_iff_f_eq_zero_l457_45755

-- Define a function f representing the curve
variable (f : ℝ → ℝ → ℝ)

-- Define a point P
variable (x₀ y₀ : ℝ)

-- Theorem stating the necessary and sufficient condition
theorem point_on_curve_iff_f_eq_zero :
  (∃ (x y : ℝ), f x y = 0 ∧ x = x₀ ∧ y = y₀) ↔ f x₀ y₀ = 0 := by sorry

end NUMINAMATH_CALUDE_point_on_curve_iff_f_eq_zero_l457_45755


namespace NUMINAMATH_CALUDE_specific_ellipse_area_l457_45783

/-- An ellipse with given major axis endpoints and a point on its curve -/
structure Ellipse where
  major_axis_end1 : ℝ × ℝ
  major_axis_end2 : ℝ × ℝ
  point_on_curve : ℝ × ℝ

/-- Calculate the area of the ellipse -/
def ellipse_area (e : Ellipse) : ℝ := sorry

/-- Theorem: The area of the specific ellipse is 50π -/
theorem specific_ellipse_area :
  let e : Ellipse := {
    major_axis_end1 := (2, -3),
    major_axis_end2 := (22, -3),
    point_on_curve := (20, 0)
  }
  ellipse_area e = 50 * Real.pi := by sorry

end NUMINAMATH_CALUDE_specific_ellipse_area_l457_45783


namespace NUMINAMATH_CALUDE_hundredths_place_of_seven_twentyfifths_l457_45796

theorem hundredths_place_of_seven_twentyfifths : ∃ (n : ℕ), (7 : ℚ) / 25 = (n + 28) / 100 ∧ n % 10 = 0 :=
sorry

end NUMINAMATH_CALUDE_hundredths_place_of_seven_twentyfifths_l457_45796


namespace NUMINAMATH_CALUDE_simplify_expression_l457_45703

theorem simplify_expression (x : ℝ) : x * (4 * x^2 - 3) - 6 * (x^2 - 3*x + 8) = 4 * x^3 - 6 * x^2 + 15 * x - 48 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l457_45703


namespace NUMINAMATH_CALUDE_toms_trip_speed_l457_45700

/-- Proves that given the conditions of Tom's trip, his speed during the first part was 20 mph -/
theorem toms_trip_speed : 
  ∀ (v : ℝ),
  (v > 0) →
  (50 / v + 1 > 0) →
  (100 / (50 / v + 1) = 28.571428571428573) →
  v = 20 := by
  sorry

end NUMINAMATH_CALUDE_toms_trip_speed_l457_45700


namespace NUMINAMATH_CALUDE_solve_unknown_months_l457_45772

/-- Represents the grazing arrangement for a milkman -/
structure GrazingArrangement where
  cows : ℕ
  months : ℕ

/-- Represents the rental arrangement for the pasture -/
structure RentalArrangement where
  milkmenCount : ℕ
  totalRent : ℕ
  arrangements : List GrazingArrangement
  unknownMonths : ℕ
  knownRentShare : ℕ

def pasture : RentalArrangement := {
  milkmenCount := 4,
  totalRent := 6500,
  arrangements := [
    { cows := 24, months := 3 },  -- A
    { cows := 10, months := 0 },  -- B (months unknown)
    { cows := 35, months := 4 },  -- C
    { cows := 21, months := 3 }   -- D
  ],
  unknownMonths := 0,  -- We'll solve for this
  knownRentShare := 1440  -- A's share
}

theorem solve_unknown_months (p : RentalArrangement) : p.unknownMonths = 5 :=
  sorry

end NUMINAMATH_CALUDE_solve_unknown_months_l457_45772


namespace NUMINAMATH_CALUDE_easter_egg_hunt_l457_45736

theorem easter_egg_hunt (baskets : ℕ) (eggs_per_basket : ℕ) (eggs_per_person : ℕ) 
  (shondas_kids : ℕ) (friends : ℕ) (shonda : ℕ) :
  baskets = 15 →
  eggs_per_basket = 12 →
  eggs_per_person = 9 →
  shondas_kids = 2 →
  friends = 10 →
  shonda = 1 →
  (baskets * eggs_per_basket) / eggs_per_person - (shondas_kids + friends + shonda) = 7 :=
by sorry

end NUMINAMATH_CALUDE_easter_egg_hunt_l457_45736


namespace NUMINAMATH_CALUDE_fraction_reciprocal_l457_45714

theorem fraction_reciprocal (a b : ℚ) (h : a ≠ b) :
  let c := -(a + b)
  (a + c) / (b + c) = b / a := by
sorry

end NUMINAMATH_CALUDE_fraction_reciprocal_l457_45714


namespace NUMINAMATH_CALUDE_circle_configuration_theorem_l457_45784

/-- A configuration of circles as described in the problem -/
structure CircleConfiguration where
  R : ℝ  -- Radius of the semicircle
  r : ℝ  -- Radius of circle O
  r₁ : ℝ  -- Radius of circle O₁
  r₂ : ℝ  -- Radius of circle O₂
  h_positive_R : 0 < R
  h_positive_r : 0 < r
  h_positive_r₁ : 0 < r₁
  h_positive_r₂ : 0 < r₂
  h_tangent_O : r < R  -- O is tangent to the semicircle and its diameter
  h_tangent_O₁ : r₁ < R  -- O₁ is tangent to the semicircle and its diameter
  h_tangent_O₂ : r₂ < R  -- O₂ is tangent to the semicircle and its diameter
  h_tangent_O₁_O : r + r₁ < R  -- O₁ is tangent to O
  h_tangent_O₂_O : r + r₂ < R  -- O₂ is tangent to O

/-- The main theorem to be proved -/
theorem circle_configuration_theorem (c : CircleConfiguration) :
  1 / Real.sqrt c.r₁ + 1 / Real.sqrt c.r₂ = 2 * Real.sqrt 2 / Real.sqrt c.r :=
sorry

end NUMINAMATH_CALUDE_circle_configuration_theorem_l457_45784


namespace NUMINAMATH_CALUDE_prob_green_ball_is_five_ninths_l457_45778

structure Container where
  red : ℕ
  green : ℕ

def containers : List Container := [
  ⟨10, 5⟩,
  ⟨3, 6⟩,
  ⟨4, 8⟩
]

def total_balls (c : Container) : ℕ := c.red + c.green

def prob_green (c : Container) : ℚ :=
  c.green / (total_balls c)

theorem prob_green_ball_is_five_ninths :
  (List.sum (containers.map (λ c => (1 : ℚ) / containers.length * prob_green c))) = 5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_prob_green_ball_is_five_ninths_l457_45778


namespace NUMINAMATH_CALUDE_cost_calculation_l457_45725

/-- The total cost of buying apples and bananas -/
def total_cost (a b : ℝ) : ℝ := 2 * a + 3 * b

/-- Theorem: The total cost of buying 2 kg of apples at 'a' yuan/kg and 3 kg of bananas at 'b' yuan/kg is (2a + 3b) yuan -/
theorem cost_calculation (a b : ℝ) :
  total_cost a b = 2 * a + 3 * b := by
  sorry

end NUMINAMATH_CALUDE_cost_calculation_l457_45725


namespace NUMINAMATH_CALUDE_base_eight_subtraction_l457_45734

/-- Represents a number in base 8 --/
def BaseEight : Type := ℕ

/-- Convert a base 8 number to decimal --/
def to_decimal (n : BaseEight) : ℕ := sorry

/-- Convert a decimal number to base 8 --/
def to_base_eight (n : ℕ) : BaseEight := sorry

/-- Subtraction in base 8 --/
def base_eight_sub (a b : BaseEight) : BaseEight := 
  to_base_eight (to_decimal a - to_decimal b)

theorem base_eight_subtraction : 
  base_eight_sub (to_base_eight 42) (to_base_eight 25) = to_base_eight 17 := by sorry

end NUMINAMATH_CALUDE_base_eight_subtraction_l457_45734


namespace NUMINAMATH_CALUDE_percentage_sixth_graders_combined_l457_45785

theorem percentage_sixth_graders_combined (annville_total : ℕ) (cleona_total : ℕ)
  (annville_sixth_percent : ℚ) (cleona_sixth_percent : ℚ) :
  annville_total = 100 →
  cleona_total = 200 →
  annville_sixth_percent = 11 / 100 →
  cleona_sixth_percent = 17 / 100 →
  let annville_sixth := (annville_sixth_percent * annville_total : ℚ).floor
  let cleona_sixth := (cleona_sixth_percent * cleona_total : ℚ).floor
  let total_sixth := annville_sixth + cleona_sixth
  let total_students := annville_total + cleona_total
  (total_sixth : ℚ) / total_students = 15 / 100 :=
by sorry

end NUMINAMATH_CALUDE_percentage_sixth_graders_combined_l457_45785


namespace NUMINAMATH_CALUDE_two_distinct_roots_l457_45761

/-- The cubic function f(x) = x^3 - 3x + a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + a

/-- Theorem stating that f(x) has exactly two distinct roots iff a = 2/√3 -/
theorem two_distinct_roots (a : ℝ) (h : a > 0) :
  (∃ x y : ℝ, x ≠ y ∧ f a x = 0 ∧ f a y = 0 ∧ ∀ z : ℝ, f a z = 0 → z = x ∨ z = y) ↔
  a = 2 / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_two_distinct_roots_l457_45761


namespace NUMINAMATH_CALUDE_modulus_of_complex_fraction_l457_45704

theorem modulus_of_complex_fraction (i : ℂ) (h : i^2 = -1) :
  Complex.abs (2 * i / (1 - i)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fraction_l457_45704


namespace NUMINAMATH_CALUDE_angle_B_in_triangle_l457_45781

theorem angle_B_in_triangle (A B C : ℝ) (BC AC : ℝ) (h1 : BC = 6) (h2 : AC = 4) (h3 : Real.sin A = 3/4) :
  B = π/6 := by
  sorry

end NUMINAMATH_CALUDE_angle_B_in_triangle_l457_45781


namespace NUMINAMATH_CALUDE_calculation_error_exists_l457_45723

def numbers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9]

def is_valid_expression (expr : List (Bool × ℕ)) : Prop :=
  expr.map (Prod.snd) = numbers

def evaluate_expression (expr : List (Bool × ℕ)) : ℤ :=
  expr.foldl (λ acc (op, n) => if op then acc + n else acc - n) 0

theorem calculation_error_exists 
  (expr1 expr2 : List (Bool × ℕ)) 
  (h1 : is_valid_expression expr1)
  (h2 : is_valid_expression expr2)
  (h3 : Odd (evaluate_expression expr1))
  (h4 : Even (evaluate_expression expr2)) :
  ∃ expr, expr ∈ [expr1, expr2] ∧ evaluate_expression expr ≠ 33 ∧ evaluate_expression expr ≠ 32 := by
  sorry

end NUMINAMATH_CALUDE_calculation_error_exists_l457_45723


namespace NUMINAMATH_CALUDE_set_operation_result_l457_45775

def A : Set ℕ := {0, 1, 2, 4, 5, 7}
def B : Set ℕ := {1, 3, 6, 8, 9}
def C : Set ℕ := {3, 7, 8}

theorem set_operation_result : (A ∩ B) ∪ C = {1, 3, 7, 8} := by sorry

end NUMINAMATH_CALUDE_set_operation_result_l457_45775


namespace NUMINAMATH_CALUDE_marble_ratio_l457_45782

theorem marble_ratio (total : ℕ) (red : ℕ) (yellow : ℕ) 
  (h_total : total = 85)
  (h_red : red = 14)
  (h_yellow : yellow = 29) :
  (total - red - yellow) / red = 3 := by
sorry

end NUMINAMATH_CALUDE_marble_ratio_l457_45782


namespace NUMINAMATH_CALUDE_pen_count_theorem_l457_45728

theorem pen_count_theorem : ∀ (red black blue green purple : ℕ),
  red = 8 →
  black = (150 * red) / 100 →
  blue = black + 5 →
  green = blue / 2 →
  purple = 5 →
  red + black + blue + green + purple = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_pen_count_theorem_l457_45728


namespace NUMINAMATH_CALUDE_max_d_value_l457_45768

def is_valid_number (d e : Nat) : Prop :=
  d ≤ 9 ∧ e ≤ 9 ∧ (808450 + 100000 * d + e) % 45 = 0

theorem max_d_value :
  ∃ (d : Nat), is_valid_number d 2 ∧
  ∀ (d' : Nat), is_valid_number d' 2 → d' ≤ d :=
by sorry

end NUMINAMATH_CALUDE_max_d_value_l457_45768


namespace NUMINAMATH_CALUDE_horse_speed_calculation_l457_45752

/-- The speed of the bullet in feet per second -/
def bullet_speed : ℝ := 400

/-- The difference in speed between firing in the same direction as the horse
    and the opposite direction, in feet per second -/
def speed_difference : ℝ := 40

/-- The speed of the horse in feet per second -/
def horse_speed : ℝ := 20

/-- Theorem stating that given the bullet speed and speed difference,
    the horse's speed is 20 feet per second -/
theorem horse_speed_calculation :
  (bullet_speed + horse_speed) - (bullet_speed - horse_speed) = speed_difference :=
by sorry

end NUMINAMATH_CALUDE_horse_speed_calculation_l457_45752


namespace NUMINAMATH_CALUDE_greatest_integer_with_gcd_six_l457_45715

def is_target (n : ℕ) : Prop :=
  n < 150 ∧ Nat.gcd n 18 = 6

theorem greatest_integer_with_gcd_six :
  ∃ (m : ℕ), is_target m ∧ ∀ (k : ℕ), is_target k → k ≤ m :=
by
  use 144
  sorry

end NUMINAMATH_CALUDE_greatest_integer_with_gcd_six_l457_45715


namespace NUMINAMATH_CALUDE_angela_action_figures_l457_45769

theorem angela_action_figures (initial : ℕ) : 
  (initial : ℚ) * (3/4) * (2/3) = 12 → initial = 24 := by
  sorry

end NUMINAMATH_CALUDE_angela_action_figures_l457_45769


namespace NUMINAMATH_CALUDE_hyperbola_vertex_distance_l457_45764

/-- The distance between the vertices of the hyperbola x^2/64 - y^2/49 = 1 is 16 -/
theorem hyperbola_vertex_distance : 
  let h : ℝ → ℝ → Prop := λ x y => x^2/64 - y^2/49 = 1
  ∃ x₁ x₂ : ℝ, h x₁ 0 ∧ h x₂ 0 ∧ |x₁ - x₂| = 16 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_vertex_distance_l457_45764


namespace NUMINAMATH_CALUDE_prob_three_two_digit_l457_45721

/-- The number of dice being rolled -/
def num_dice : ℕ := 6

/-- The number of sides on each die -/
def num_sides : ℕ := 20

/-- The probability of rolling a two-digit number on a single die -/
def p_two_digit : ℚ := 11 / 20

/-- The probability of rolling a one-digit number on a single die -/
def p_one_digit : ℚ := 9 / 20

/-- The probability of exactly three dice showing a two-digit number when rolling 6 20-sided dice -/
theorem prob_three_two_digit : 
  (num_dice.choose 3 : ℚ) * p_two_digit ^ 3 * p_one_digit ^ 3 = 973971 / 3200000 :=
sorry

end NUMINAMATH_CALUDE_prob_three_two_digit_l457_45721


namespace NUMINAMATH_CALUDE_rebecca_hours_l457_45730

/-- Given the working hours of Thomas, Toby, and Rebecca, prove that Rebecca worked 56 hours. -/
theorem rebecca_hours :
  ∀ x : ℕ,
  (x + (2*x - 10) + (2*x - 18) = 157) →
  (2*x - 18 = 56) :=
by
  sorry

end NUMINAMATH_CALUDE_rebecca_hours_l457_45730


namespace NUMINAMATH_CALUDE_tony_temperature_l457_45751

/-- Represents the temperature change caused by an illness -/
structure Illness where
  temp_change : Int

/-- Calculates the final temperature and its relation to the fever threshold -/
def calculate_temperature (normal_temp : Int) (illnesses : List Illness) (fever_threshold : Int) :
  (Int × Int) :=
  let final_temp := normal_temp + (illnesses.map (·.temp_change)).sum
  let above_threshold := final_temp - fever_threshold
  (final_temp, above_threshold)

theorem tony_temperature :
  let normal_temp := 95
  let illness_a := Illness.mk 10
  let illness_b := Illness.mk 4
  let illness_c := Illness.mk (-2)
  let illnesses := [illness_a, illness_b, illness_c]
  let fever_threshold := 100
  calculate_temperature normal_temp illnesses fever_threshold = (107, 7) := by
  sorry

end NUMINAMATH_CALUDE_tony_temperature_l457_45751


namespace NUMINAMATH_CALUDE_fraction_1840s_eq_four_fifteenths_l457_45739

/-- The number of states admitted between 1840 and 1849 -/
def states_1840s : ℕ := 8

/-- The total number of states in Alice's collection -/
def total_states : ℕ := 30

/-- The fraction of states admitted between 1840 and 1849 out of the first 30 states -/
def fraction_1840s : ℚ := states_1840s / total_states

theorem fraction_1840s_eq_four_fifteenths : fraction_1840s = 4 / 15 := by
  sorry

end NUMINAMATH_CALUDE_fraction_1840s_eq_four_fifteenths_l457_45739


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l457_45763

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) ↔ a ∈ Set.Ioc (-2) 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l457_45763


namespace NUMINAMATH_CALUDE_shirt_price_satisfies_conditions_l457_45753

/-- The original price of a shirt, given the following conditions:
  1. Three items: shirt, pants, jacket
  2. Shirt: 25% discount, then additional 25% discount
  3. Pants: 30% discount, original price $50
  4. Jacket: two successive 20% discounts, original price $75
  5. 10% loyalty discount on total after individual discounts
  6. 15% sales tax on final price
  7. Total price paid: $150
-/
def shirt_price : ℝ :=
  let pants_price : ℝ := 50
  let jacket_price : ℝ := 75
  let pants_discount : ℝ := 0.30
  let jacket_discount : ℝ := 0.20
  let loyalty_discount : ℝ := 0.10
  let sales_tax : ℝ := 0.15
  let total_paid : ℝ := 150
  sorry

/-- Theorem stating that the calculated shirt price satisfies the given conditions -/
theorem shirt_price_satisfies_conditions :
  let S := shirt_price
  let pants_discounted := 50 * (1 - 0.30)
  let jacket_discounted := 75 * (1 - 0.20) * (1 - 0.20)
  (S * 0.75 * 0.75 + pants_discounted + jacket_discounted) * (1 - 0.10) * (1 + 0.15) = 150 := by
  sorry

end NUMINAMATH_CALUDE_shirt_price_satisfies_conditions_l457_45753


namespace NUMINAMATH_CALUDE_max_value_of_expression_max_value_achievable_l457_45791

theorem max_value_of_expression (y : ℝ) :
  y^6 / (y^12 + 3*y^9 - 9*y^6 + 27*y^3 + 81) ≤ 1/27 :=
by sorry

theorem max_value_achievable :
  ∃ y : ℝ, y^6 / (y^12 + 3*y^9 - 9*y^6 + 27*y^3 + 81) = 1/27 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_max_value_achievable_l457_45791


namespace NUMINAMATH_CALUDE_shooting_target_proof_l457_45776

theorem shooting_target_proof (p q : Prop) : 
  (¬p ∨ ¬q) ↔ (¬(p ∧ q)) :=
sorry

end NUMINAMATH_CALUDE_shooting_target_proof_l457_45776


namespace NUMINAMATH_CALUDE_f_increasing_on_interval_l457_45742

open Real

noncomputable def f (x : ℝ) : ℝ := x - 2 * sin x

theorem f_increasing_on_interval :
  ∀ x ∈ Set.Ioo (π/3) (5*π/3), 
    x ∈ Set.Ioo 0 (2*π) → 
    ∀ y ∈ Set.Ioo (π/3) (5*π/3), 
      x < y → f x < f y :=
by sorry

end NUMINAMATH_CALUDE_f_increasing_on_interval_l457_45742


namespace NUMINAMATH_CALUDE_inner_probability_is_16_25_l457_45789

/-- The size of one side of the square checkerboard -/
def boardSize : ℕ := 10

/-- The total number of squares on the checkerboard -/
def totalSquares : ℕ := boardSize * boardSize

/-- The number of squares on the perimeter of the checkerboard -/
def perimeterSquares : ℕ := 4 * boardSize - 4

/-- The number of squares not on the perimeter of the checkerboard -/
def innerSquares : ℕ := totalSquares - perimeterSquares

/-- The probability of choosing a square not on the perimeter -/
def innerProbability : ℚ := innerSquares / totalSquares

theorem inner_probability_is_16_25 : innerProbability = 16 / 25 := by
  sorry

end NUMINAMATH_CALUDE_inner_probability_is_16_25_l457_45789


namespace NUMINAMATH_CALUDE_average_problem_l457_45788

theorem average_problem (y : ℝ) (h : (15 + 24 + 32 + y) / 4 = 26) : y = 33 := by
  sorry

end NUMINAMATH_CALUDE_average_problem_l457_45788


namespace NUMINAMATH_CALUDE_dance_class_boys_count_l457_45779

theorem dance_class_boys_count :
  ∀ (girls boys : ℕ),
  girls + boys = 35 →
  4 * girls = 3 * boys →
  boys = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_dance_class_boys_count_l457_45779


namespace NUMINAMATH_CALUDE_group_size_proof_l457_45716

theorem group_size_proof (total_collection : ℚ) (paise_per_rupee : ℕ) : 
  (total_collection = 32.49) →
  (paise_per_rupee = 100) →
  ∃ n : ℕ, (n * n = total_collection * paise_per_rupee) ∧ (n = 57) :=
by sorry

end NUMINAMATH_CALUDE_group_size_proof_l457_45716


namespace NUMINAMATH_CALUDE_nancy_folders_l457_45771

-- Define the problem parameters
def initial_files : ℕ := 80
def deleted_files : ℕ := 31
def files_per_folder : ℕ := 7

-- Define the function to calculate the number of folders
def calculate_folders (initial : ℕ) (deleted : ℕ) (per_folder : ℕ) : ℕ :=
  (initial - deleted) / per_folder

-- State the theorem
theorem nancy_folders :
  calculate_folders initial_files deleted_files files_per_folder = 7 := by
  sorry

end NUMINAMATH_CALUDE_nancy_folders_l457_45771


namespace NUMINAMATH_CALUDE_triangle_inequality_l457_45760

theorem triangle_inequality (a b c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_perimeter : a + b + c = 1) :
  a^2 + b^2 + c^2 + 4*a*b*c < 1/2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l457_45760


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l457_45777

-- Problem 1
theorem problem_1 (x : ℝ) (h : x > 0) (eq : Real.sqrt x + 1 / Real.sqrt x = 3) : 
  x + 1 / x = 7 := by sorry

-- Problem 2
theorem problem_2 : 
  (Real.log 3 / Real.log 4 + Real.log 3 / Real.log 8) * (Real.log 2 / Real.log 3 + Real.log 8 / Real.log 9) = 25 / 12 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l457_45777


namespace NUMINAMATH_CALUDE_willie_stickers_l457_45743

/-- The number of stickers Willie ends up with after giving some away -/
def stickers_left (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

/-- Theorem: Willie ends up with 29 stickers -/
theorem willie_stickers : stickers_left 36 7 = 29 := by
  sorry

end NUMINAMATH_CALUDE_willie_stickers_l457_45743


namespace NUMINAMATH_CALUDE_largest_divisor_of_n_l457_45793

theorem largest_divisor_of_n (n : ℕ) (h1 : n > 0) (h2 : 72 ∣ n^2) : ∃ d : ℕ, d > 0 ∧ d ∣ n ∧ ∀ k : ℕ, k > 0 → k ∣ n → k ≤ d := by
  sorry

end NUMINAMATH_CALUDE_largest_divisor_of_n_l457_45793


namespace NUMINAMATH_CALUDE_gunther_free_time_l457_45780

def cleaning_time (vacuum_time dust_time mop_time brush_time_per_cat num_cats : ℕ) : ℕ :=
  vacuum_time + dust_time + mop_time + brush_time_per_cat * num_cats

theorem gunther_free_time 
  (free_time : ℕ) 
  (vacuum_time : ℕ)
  (dust_time : ℕ)
  (mop_time : ℕ)
  (brush_time_per_cat : ℕ)
  (num_cats : ℕ)
  (h1 : free_time = 3 * 60)
  (h2 : vacuum_time = 45)
  (h3 : dust_time = 60)
  (h4 : mop_time = 30)
  (h5 : brush_time_per_cat = 5)
  (h6 : num_cats = 3) :
  free_time - cleaning_time vacuum_time dust_time mop_time brush_time_per_cat num_cats = 30 :=
by sorry

end NUMINAMATH_CALUDE_gunther_free_time_l457_45780


namespace NUMINAMATH_CALUDE_moms_dimes_l457_45797

/-- Given the initial number of dimes, the number of dimes given by dad, and the final number of dimes,
    proves that the number of dimes given by mom is 4. -/
theorem moms_dimes (initial : ℕ) (from_dad : ℕ) (final : ℕ)
  (h1 : initial = 7)
  (h2 : from_dad = 8)
  (h3 : final = 19) :
  final - (initial + from_dad) = 4 := by
  sorry

end NUMINAMATH_CALUDE_moms_dimes_l457_45797


namespace NUMINAMATH_CALUDE_equation_solution_l457_45798

theorem equation_solution : 
  {x : ℝ | Real.sqrt ((1 + Real.sqrt 2) ^ x) + Real.sqrt ((1 - Real.sqrt 2) ^ x) = 3} = {2, -2} :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l457_45798


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l457_45701

theorem quadratic_no_real_roots : ∀ x : ℝ, x^2 - 2*x + 3 ≠ 0 := by
  sorry

#check quadratic_no_real_roots

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l457_45701


namespace NUMINAMATH_CALUDE_system_1_solution_system_2_solution_l457_45745

-- System 1
theorem system_1_solution :
  ∃ (x y : ℝ), x - y = 3 ∧ x = 3 * y - 1 ∧ x = 5 ∧ y = 2 := by
  sorry

-- System 2
theorem system_2_solution :
  ∃ (x y : ℝ), 2 * x + 3 * y = -1 ∧ 3 * x - 2 * y = 18 ∧ x = 4 ∧ y = -3 := by
  sorry

end NUMINAMATH_CALUDE_system_1_solution_system_2_solution_l457_45745


namespace NUMINAMATH_CALUDE_negation_of_exp_greater_than_x_l457_45719

theorem negation_of_exp_greater_than_x :
  (¬ ∀ x : ℝ, Real.exp x > x) ↔ (∃ x : ℝ, Real.exp x ≤ x) := by sorry

end NUMINAMATH_CALUDE_negation_of_exp_greater_than_x_l457_45719


namespace NUMINAMATH_CALUDE_contour_bar_chart_judges_relationship_l457_45722

/-- Represents a method for judging the relationship between categorical variables -/
inductive IndependenceTestMethod
  | Residuals
  | ContourBarChart
  | HypothesisTesting
  | Other

/-- Defines the property of being able to roughly judge the relationship between categorical variables -/
def can_roughly_judge_relationship (method : IndependenceTestMethod) : Prop :=
  match method with
  | IndependenceTestMethod.ContourBarChart => True
  | _ => False

/-- Theorem stating that a contour bar chart can be used to roughly judge the relationship between categorical variables in an independence test -/
theorem contour_bar_chart_judges_relationship :
  can_roughly_judge_relationship IndependenceTestMethod.ContourBarChart :=
sorry

end NUMINAMATH_CALUDE_contour_bar_chart_judges_relationship_l457_45722


namespace NUMINAMATH_CALUDE_statement_1_incorrect_statement_4_incorrect_l457_45712

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (contains : Plane → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (perpendicularToPlane : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- Statement 1
theorem statement_1_incorrect 
  (α : Plane) (l m n : Line) : 
  ¬(∀ (α : Plane) (l m n : Line), 
    contains α m → contains α n → perpendicular l m → perpendicular l n → 
    perpendicularToPlane l α) := 
by sorry

-- Statement 4
theorem statement_4_incorrect 
  (α : Plane) (l m n : Line) : 
  ¬(∀ (α : Plane) (l m n : Line), 
    contains α m → perpendicularToPlane n α → perpendicular l n → 
    parallel l m) := 
by sorry

end NUMINAMATH_CALUDE_statement_1_incorrect_statement_4_incorrect_l457_45712


namespace NUMINAMATH_CALUDE_ratio_of_numbers_l457_45794

theorem ratio_of_numbers (x y : ℝ) (h1 : x + y = 14) (h2 : y = 3.5) (h3 : x > y) :
  x / y = 3 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_numbers_l457_45794


namespace NUMINAMATH_CALUDE_sin_90_degrees_l457_45727

theorem sin_90_degrees : Real.sin (π / 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_90_degrees_l457_45727
