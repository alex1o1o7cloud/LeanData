import Mathlib

namespace child_b_share_child_b_share_is_552_l2867_286777

/-- Calculates the share of child B given the total amount, tax rate, interest rate, and distribution ratio. -/
theorem child_b_share (total_amount : ℝ) (tax_rate : ℝ) (interest_rate : ℝ) (ratio_a ratio_b ratio_c : ℕ) : ℝ :=
  let tax := total_amount * tax_rate
  let interest := total_amount * interest_rate
  let remaining_amount := total_amount - (tax + interest)
  let total_parts := ratio_a + ratio_b + ratio_c
  let part_value := remaining_amount / total_parts
  ratio_b * part_value

/-- Proves that given the specific conditions, B's share is $552. -/
theorem child_b_share_is_552 : 
  child_b_share 1800 0.05 0.03 2 3 4 = 552 := by
  sorry

end child_b_share_child_b_share_is_552_l2867_286777


namespace second_polygon_sides_l2867_286701

/-- Given two regular polygons with the same perimeter, where one has 50 sides
    and a side length three times as long as the other, prove that the number
    of sides of the second polygon is 150. -/
theorem second_polygon_sides (s : ℝ) (n : ℕ) : s > 0 →
  50 * (3 * s) = n * s → n = 150 := by sorry

end second_polygon_sides_l2867_286701


namespace fertilizer_prices_l2867_286708

/-- Represents the price per ton of fertilizer A -/
def price_A : ℝ := sorry

/-- Represents the price per ton of fertilizer B -/
def price_B : ℝ := sorry

/-- The price difference between fertilizer A and B is $100 -/
axiom price_difference : price_A = price_B + 100

/-- The total cost of 2 tons of fertilizer A and 1 ton of fertilizer B is $1700 -/
axiom total_cost : 2 * price_A + price_B = 1700

theorem fertilizer_prices :
  price_A = 600 ∧ price_B = 500 := by sorry

end fertilizer_prices_l2867_286708


namespace abc_mod_nine_l2867_286792

theorem abc_mod_nine (a b c : ℕ) (ha : a < 9) (hb : b < 9) (hc : c < 9)
  (h1 : (a + 2*b + 3*c) % 9 = 0)
  (h2 : (2*a + 3*b + c) % 9 = 3)
  (h3 : (3*a + b + 2*c) % 9 = 8) :
  (a * b * c) % 9 = 6 := by
  sorry

end abc_mod_nine_l2867_286792


namespace greatest_n_value_exists_n_value_l2867_286753

theorem greatest_n_value (n : ℤ) (h : 101 * n^2 ≤ 12100) : n ≤ 10 := by
  sorry

theorem exists_n_value : ∃ (n : ℤ), 101 * n^2 ≤ 12100 ∧ n = 10 := by
  sorry

end greatest_n_value_exists_n_value_l2867_286753


namespace sqrt_meaningful_range_l2867_286702

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 6 + x) ↔ x ≥ -6 := by
  sorry

end sqrt_meaningful_range_l2867_286702


namespace rent_increase_percentage_l2867_286781

/-- Proves that the percentage increase in rent for one friend is 16% given the conditions of the problem -/
theorem rent_increase_percentage (num_friends : ℕ) (initial_avg : ℝ) (new_avg : ℝ) (initial_rent : ℝ) : 
  num_friends = 4 →
  initial_avg = 800 →
  new_avg = 850 →
  initial_rent = 1250 →
  let total_initial := initial_avg * num_friends
  let new_rent := (new_avg * num_friends) - (total_initial - initial_rent)
  let percentage_increase := (new_rent - initial_rent) / initial_rent * 100
  percentage_increase = 16 := by
sorry

end rent_increase_percentage_l2867_286781


namespace additional_money_needed_l2867_286713

/-- The amount of money Cory has initially -/
def initial_money : ℚ := 20

/-- The cost of one pack of candies -/
def candy_pack_cost : ℚ := 49

/-- The number of candy packs Cory wants to buy -/
def num_packs : ℕ := 2

/-- Theorem: Given Cory's initial money and the cost of candy packs,
    the additional amount needed to buy two packs is $78.00 -/
theorem additional_money_needed :
  (candy_pack_cost * num_packs : ℚ) - initial_money = 78 := by
  sorry

end additional_money_needed_l2867_286713


namespace rectangle_longer_side_l2867_286788

theorem rectangle_longer_side (r : ℝ) (h1 : r = 6) : ∃ L : ℝ,
  (L * (2 * r) = 3 * (π * r^2)) ∧ L = 9 * π := by
  sorry

end rectangle_longer_side_l2867_286788


namespace x_plus_y_equals_eight_l2867_286787

theorem x_plus_y_equals_eight (x y : ℝ) 
  (h1 : |x| - x + y = 8) 
  (h2 : x + |y| + y = 16) : 
  x + y = 8 := by sorry

end x_plus_y_equals_eight_l2867_286787


namespace cross_product_equals_l2867_286741

def vector1 : ℝ × ℝ × ℝ := (3, -4, 5)
def vector2 : ℝ × ℝ × ℝ := (-2, 7, 1)

def cross_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (a, b, c) := v1
  let (d, e, f) := v2
  (b * f - c * e, c * d - a * f, a * e - b * d)

theorem cross_product_equals : cross_product vector1 vector2 = (-39, -13, 13) := by
  sorry

end cross_product_equals_l2867_286741


namespace train_speed_calculation_l2867_286751

/-- Given a train that crosses a pole in a certain time, calculate its speed in kmph. -/
theorem train_speed_calculation (train_length : Real) (crossing_time : Real) :
  train_length = 800.064 →
  crossing_time = 18 →
  (train_length / 1000) / (crossing_time / 3600) = 160.0128 := by
  sorry

#check train_speed_calculation

end train_speed_calculation_l2867_286751


namespace order_of_xyz_l2867_286761

-- Define the variables and their relationships
theorem order_of_xyz (a b c d : ℝ) 
  (h_order : a > b ∧ b > c ∧ c > d ∧ d > 0) 
  (x : ℝ) (hx : x = Real.sqrt (a * b) + Real.sqrt (c * d))
  (y : ℝ) (hy : y = Real.sqrt (a * c) + Real.sqrt (b * d))
  (z : ℝ) (hz : z = Real.sqrt (a * d) + Real.sqrt (b * c)) :
  x > y ∧ y > z :=
by sorry

end order_of_xyz_l2867_286761


namespace base8_563_to_base3_l2867_286744

/-- Converts a base 8 number to base 10 --/
def base8ToBase10 (n : Nat) : Nat :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds * 8^2 + tens * 8^1 + ones * 8^0

/-- Converts a base 10 number to base 3 --/
def base10ToBase3 (n : Nat) : List Nat :=
  sorry  -- Implementation details omitted

theorem base8_563_to_base3 :
  base10ToBase3 (base8ToBase10 563) = [1, 1, 1, 2, 2, 0] := by
  sorry

end base8_563_to_base3_l2867_286744


namespace vector_addition_l2867_286728

theorem vector_addition (a b : Fin 2 → ℝ) 
  (ha : a = ![2, 1]) 
  (hb : b = ![1, 3]) : 
  a + b = ![3, 4] := by
  sorry

end vector_addition_l2867_286728


namespace son_age_l2867_286722

theorem son_age (father_age son_age : ℕ) : 
  father_age = son_age + 22 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 20 := by
sorry

end son_age_l2867_286722


namespace billion_difference_value_l2867_286721

/-- Arnaldo's definition of a billion -/
def arnaldo_billion : ℕ := 1000000 * 1000000

/-- Correct definition of a billion -/
def correct_billion : ℕ := 1000 * 1000000

/-- The difference between Arnaldo's definition and the correct definition -/
def billion_difference : ℕ := arnaldo_billion - correct_billion

theorem billion_difference_value : billion_difference = 999000000000 := by
  sorry

end billion_difference_value_l2867_286721


namespace integer_root_values_l2867_286760

def has_integer_root (b : ℤ) : Prop :=
  ∃ x : ℤ, x^3 + 2*x^2 + b*x + 8 = 0

theorem integer_root_values :
  {b : ℤ | has_integer_root b} = {-81, -26, -19, -12, -11, 4, 9, 47} := by sorry

end integer_root_values_l2867_286760


namespace jeff_new_cabinet_counters_l2867_286767

/-- Calculates the number of counters over which new cabinets were installed --/
def counters_with_new_cabinets (initial_cabinets : ℕ) (cabinets_per_new_counter : ℕ) (additional_cabinets : ℕ) (total_cabinets : ℕ) : ℕ :=
  (total_cabinets - initial_cabinets - additional_cabinets) / cabinets_per_new_counter

/-- Proves that Jeff installed new cabinets over 9 counters --/
theorem jeff_new_cabinet_counters :
  let initial_cabinets := 3
  let cabinets_per_new_counter := 2
  let additional_cabinets := 5
  let total_cabinets := 26
  counters_with_new_cabinets initial_cabinets cabinets_per_new_counter additional_cabinets total_cabinets = 9 := by
  sorry

end jeff_new_cabinet_counters_l2867_286767


namespace standard_form_of_negative_r_l2867_286749

/-- Converts a polar coordinate point to its standard form where r > 0 and 0 ≤ θ < 2π -/
def standardPolarForm (r : ℝ) (θ : ℝ) : ℝ × ℝ :=
  sorry

theorem standard_form_of_negative_r :
  let original : ℝ × ℝ := (-3, π/6)
  let standard : ℝ × ℝ := standardPolarForm original.1 original.2
  standard = (3, 7*π/6) ∧ standard.1 > 0 ∧ 0 ≤ standard.2 ∧ standard.2 < 2*π :=
by sorry

end standard_form_of_negative_r_l2867_286749


namespace sum_first_two_terms_l2867_286794

/-- A geometric sequence with third term 12 and fourth term 18 -/
def GeometricSequence (a : ℕ → ℚ) : Prop :=
  ∃ (q : ℚ), q ≠ 0 ∧ (∀ n, a (n + 1) = a n * q) ∧ a 3 = 12 ∧ a 4 = 18

/-- The sum of the first and second terms of the geometric sequence is 40/3 -/
theorem sum_first_two_terms (a : ℕ → ℚ) (h : GeometricSequence a) :
  a 1 + a 2 = 40 / 3 := by
  sorry

end sum_first_two_terms_l2867_286794


namespace sum_of_three_fourth_powers_not_end_2019_l2867_286791

theorem sum_of_three_fourth_powers_not_end_2019 :
  ∀ a b c : ℤ, ¬ (∃ k : ℤ, a^4 + b^4 + c^4 = 10000 * k + 2019) :=
by sorry

end sum_of_three_fourth_powers_not_end_2019_l2867_286791


namespace count_valid_numbers_l2867_286739

/-- A three-digit number composed of distinct digits -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  distinct : hundreds ≠ tens ∧ tens ≠ ones ∧ hundreds ≠ ones
  valid_range : hundreds ∈ Finset.range 10 ∧ tens ∈ Finset.range 10 ∧ ones ∈ Finset.range 10

/-- Check if one digit is the average of the other two -/
def has_average_digit (n : ThreeDigitNumber) : Prop :=
  2 * n.hundreds = n.tens + n.ones ∨
  2 * n.tens = n.hundreds + n.ones ∨
  2 * n.ones = n.hundreds + n.tens

/-- Check if the sum of digits is divisible by 3 -/
def sum_divisible_by_three (n : ThreeDigitNumber) : Prop :=
  (n.hundreds + n.tens + n.ones) % 3 = 0

/-- The set of all valid three-digit numbers satisfying the conditions -/
def valid_numbers : Finset ThreeDigitNumber :=
  sorry

theorem count_valid_numbers : valid_numbers.card = 160 := by
  sorry

end count_valid_numbers_l2867_286739


namespace polynomial_sum_l2867_286799

def is_monic_degree_4 (p : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, p x = x^4 + a*x^3 + b*x^2 + c*x + d

theorem polynomial_sum (p : ℝ → ℝ) :
  is_monic_degree_4 p →
  p 1 = 17 →
  p 2 = 38 →
  p 3 = 63 →
  p 0 + p 4 = 68 :=
by
  sorry


end polynomial_sum_l2867_286799


namespace quadratic_roots_opposite_l2867_286790

theorem quadratic_roots_opposite (k : ℝ) : 
  (∃ x y : ℝ, x^2 + (k-2)*x - 1 = 0 ∧ y^2 + (k-2)*y - 1 = 0 ∧ x = -y) → k = 2 := by
  sorry

end quadratic_roots_opposite_l2867_286790


namespace compute_d_l2867_286716

-- Define the polynomial
def f (c d : ℚ) (x : ℝ) : ℝ := x^3 + c*x^2 + d*x - 36

-- State the theorem
theorem compute_d (c : ℚ) :
  ∃ d : ℚ, f c d (3 + Real.sqrt 2) = 0 → d = -23 - 6/7 :=
by sorry

end compute_d_l2867_286716


namespace units_digit_of_65_plus_37_in_octal_l2867_286725

/-- Represents a number in base 8 --/
def OctalNumber := ℕ

/-- Converts a natural number to its octal representation --/
def toOctal (n : ℕ) : OctalNumber :=
  sorry

/-- Adds two octal numbers --/
def octalAdd (a b : OctalNumber) : OctalNumber :=
  sorry

/-- Gets the units digit of an octal number --/
def unitsDigit (n : OctalNumber) : ℕ :=
  sorry

/-- Theorem: The units digit of 65₈ + 37₈ in base 8 is 4 --/
theorem units_digit_of_65_plus_37_in_octal :
  unitsDigit (octalAdd (toOctal 65) (toOctal 37)) = 4 :=
sorry

end units_digit_of_65_plus_37_in_octal_l2867_286725


namespace seating_arrangement_count_l2867_286778

/-- Represents a circular table with chairs -/
structure CircularTable :=
  (num_chairs : ℕ)

/-- Represents a group of married couples -/
structure MarriedCouples :=
  (num_couples : ℕ)

/-- Represents the constraints for seating arrangements -/
structure SeatingConstraints :=
  (alternate_gender : Bool)
  (no_adjacent_spouses : Bool)
  (no_opposite_spouses : Bool)

/-- Calculates the number of valid seating arrangements -/
noncomputable def count_seating_arrangements (table : CircularTable) (couples : MarriedCouples) (constraints : SeatingConstraints) : ℕ :=
  sorry

/-- The main theorem to be proved -/
theorem seating_arrangement_count :
  ∀ (table : CircularTable) (couples : MarriedCouples) (constraints : SeatingConstraints),
    table.num_chairs = 10 →
    couples.num_couples = 5 →
    constraints.alternate_gender = true →
    constraints.no_adjacent_spouses = true →
    constraints.no_opposite_spouses = true →
    count_seating_arrangements table couples constraints = 480 :=
by
  sorry

end seating_arrangement_count_l2867_286778


namespace great_8_teams_l2867_286711

-- Define the number of teams
def n : ℕ := sorry

-- Define the total number of games
def total_games : ℕ := 36

-- Theorem stating the conditions and the result to be proven
theorem great_8_teams :
  (∀ (i j : ℕ), i < n → j < n → i ≠ j → ∃! (game : ℕ), game < total_games) ∧
  (n * (n - 1) / 2 = total_games) →
  n = 9 := by sorry

end great_8_teams_l2867_286711


namespace hyperbola_equation_l2867_286783

-- Define the hyperbola
def hyperbola (x y : ℝ) := y^2 - x^2 = 2

-- Define the foci
def foci : Set (ℝ × ℝ) := {(0, 2), (0, -2)}

-- Define the asymptotes
def asymptotes (x y : ℝ) := x^2/3 - y^2/3 = 1

-- Theorem statement
theorem hyperbola_equation :
  ∀ (x y : ℝ),
  (∃ (f : ℝ × ℝ), f ∈ foci) →
  (∀ (x' y' : ℝ), asymptotes x' y' ↔ asymptotes x y) →
  hyperbola x y :=
sorry

end hyperbola_equation_l2867_286783


namespace four_number_sequence_l2867_286706

/-- Given four real numbers satisfying specific sequence and sum conditions, 
    prove they are one of two specific quadruples -/
theorem four_number_sequence (a b c d : ℝ) 
  (geom_seq : b / c = c / a)  -- a, b, c form a geometric sequence
  (geom_sum : a + b + c = 19)
  (arith_seq : b - c = c - d)  -- b, c, d form an arithmetic sequence
  (arith_sum : b + c + d = 12) :
  ((a, b, c, d) = (25, -10, 4, 18)) ∨ ((a, b, c, d) = (9, 6, 4, 2)) := by
  sorry

end four_number_sequence_l2867_286706


namespace mary_baseball_cards_l2867_286762

theorem mary_baseball_cards 
  (promised_to_fred : ℝ) 
  (bought : ℝ) 
  (left_after_giving : ℝ) 
  (h1 : promised_to_fred = 26.0)
  (h2 : bought = 40.0)
  (h3 : left_after_giving = 32.0) :
  ∃ initial : ℝ, initial = 18.0 ∧ 
    (initial + bought - promised_to_fred = left_after_giving) :=
by sorry

end mary_baseball_cards_l2867_286762


namespace x_value_l2867_286774

theorem x_value (x : ℚ) (h : 1/4 - 1/6 = 4/x) : x = 48 := by
  sorry

end x_value_l2867_286774


namespace parallel_condition_l2867_286717

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism relations
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)

-- Define the subset relation for lines and planes
variable (subset : Line → Plane → Prop)

-- Theorem statement
theorem parallel_condition (l m : Line) (α : Plane)
  (h1 : ¬ subset l α)
  (h2 : subset m α) :
  (∀ l m, parallel_lines l m → parallel_line_plane l α) ∧
  (∃ l m, parallel_line_plane l α ∧ ¬ parallel_lines l m) :=
sorry

end parallel_condition_l2867_286717


namespace exactly_two_pairs_exist_l2867_286752

-- Define the type for a pair of real numbers
def RealPair := ℝ × ℝ

-- Define a function to check if two lines are identical
def are_lines_identical (b c : ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ 
    (2 = k * c) ∧ 
    (3 * b = k * 4) ∧ 
    (c = k * 16)

-- Define the set of pairs (b, c) that make the lines identical
def identical_line_pairs : Set RealPair :=
  {p : RealPair | are_lines_identical p.1 p.2}

-- Theorem statement
theorem exactly_two_pairs_exist : 
  ∃ (p₁ p₂ : RealPair), p₁ ≠ p₂ ∧ 
    p₁ ∈ identical_line_pairs ∧ 
    p₂ ∈ identical_line_pairs ∧ 
    ∀ (p : RealPair), p ∈ identical_line_pairs → p = p₁ ∨ p = p₂ :=
  sorry

end exactly_two_pairs_exist_l2867_286752


namespace A_inter_B_eq_A_l2867_286776

-- Define the sets A and B
def A : Set ℝ := {x | 0 < x ∧ x ≤ 2}
def B : Set ℝ := {x | x ≤ 3}

-- Theorem statement
theorem A_inter_B_eq_A : A ∩ B = A := by sorry

end A_inter_B_eq_A_l2867_286776


namespace johns_remaining_budget_l2867_286764

/-- Calculates the remaining budget after a purchase -/
def remaining_budget (initial : ℚ) (spent : ℚ) : ℚ :=
  initial - spent

/-- Proves that given an initial budget of $999.00 and a purchase of $165.00, the remaining amount is $834.00 -/
theorem johns_remaining_budget :
  remaining_budget 999 165 = 834 := by
  sorry

end johns_remaining_budget_l2867_286764


namespace non_defective_products_percentage_l2867_286769

/-- Represents a machine in the factory -/
structure Machine where
  production_percentage : ℝ
  defective_percentage : ℝ

/-- The factory setup -/
def factory : List Machine := [
  ⟨0.25, 0.02⟩,  -- m1
  ⟨0.35, 0.04⟩,  -- m2
  ⟨0.40, 0.05⟩   -- m3
]

/-- Calculate the percentage of non-defective products -/
def non_defective_percentage (machines : List Machine) : ℝ :=
  1 - (machines.map (λ m => m.production_percentage * m.defective_percentage)).sum

/-- Theorem stating the percentage of non-defective products -/
theorem non_defective_products_percentage :
  non_defective_percentage factory = 0.961 := by
  sorry

#eval non_defective_percentage factory

end non_defective_products_percentage_l2867_286769


namespace fraction_equality_l2867_286748

theorem fraction_equality (w x y : ℚ) 
  (h1 : w / y = 3 / 4)
  (h2 : (x + y) / y = 13 / 4) :
  w / x = 1 / 3 := by
sorry

end fraction_equality_l2867_286748


namespace M_intersect_N_equals_unit_interval_l2867_286798

-- Define the sets M and N
def M : Set ℝ := {x | ∃ y, y = Real.log (2*x - x^2)}
def N : Set ℝ := {x | x^2 + 2*x - 3 ≥ 0}

-- State the theorem
theorem M_intersect_N_equals_unit_interval :
  M ∩ N = {x | 1 ≤ x ∧ x < 2} := by sorry

end M_intersect_N_equals_unit_interval_l2867_286798


namespace perfect_square_factors_of_3780_l2867_286747

def prime_factorization (n : ℕ) : List (ℕ × ℕ) := sorry

def is_perfect_square (n : ℕ) : Prop := sorry

def count_perfect_square_factors (n : ℕ) : ℕ := sorry

theorem perfect_square_factors_of_3780 :
  let factorization := prime_factorization 3780
  (factorization = [(2, 2), (3, 3), (5, 1), (7, 2)]) →
  count_perfect_square_factors 3780 = 8 := by sorry

end perfect_square_factors_of_3780_l2867_286747


namespace fraction_equality_l2867_286789

theorem fraction_equality (x y : ℝ) (h : x / y = 2) : (x - y) / x = 1 / 2 := by
  sorry

end fraction_equality_l2867_286789


namespace elberta_money_l2867_286779

theorem elberta_money (granny_smith : ℕ) (elberta anjou : ℕ) : 
  granny_smith = 72 →
  elberta = anjou + 5 →
  anjou = granny_smith / 4 →
  elberta = 23 := by
sorry

end elberta_money_l2867_286779


namespace cucumber_water_percentage_l2867_286736

/-- Calculates the new water percentage in cucumbers after evaporation -/
theorem cucumber_water_percentage
  (initial_weight : ℝ)
  (initial_water_percentage : ℝ)
  (final_weight : ℝ)
  (h1 : initial_weight = 100)
  (h2 : initial_water_percentage = 99)
  (h3 : final_weight = 20)
  : (final_weight - (initial_weight * (1 - initial_water_percentage / 100))) / final_weight * 100 = 95 := by
  sorry

end cucumber_water_percentage_l2867_286736


namespace guard_distance_proof_l2867_286786

/-- Calculates the total distance walked by a guard around a rectangular warehouse -/
def total_distance_walked (length width : ℕ) (total_circles skipped_circles : ℕ) : ℕ :=
  2 * (length + width) * (total_circles - skipped_circles)

/-- Proves that the guard walks 16000 feet given the specific conditions -/
theorem guard_distance_proof :
  total_distance_walked 600 400 10 2 = 16000 := by
  sorry

end guard_distance_proof_l2867_286786


namespace fraction_as_power_series_l2867_286705

theorem fraction_as_power_series :
  ∃ (a : ℕ → ℚ), (9 : ℚ) / 10 = (5 : ℚ) / 6 + ∑' n, a n / (6 ^ (n + 2)) :=
by sorry

end fraction_as_power_series_l2867_286705


namespace shirt_final_price_l2867_286785

/-- The final price of a shirt after two successive discounts --/
theorem shirt_final_price (list_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : 
  list_price = 150 → 
  discount1 = 19.954259576901087 →
  discount2 = 12.55 →
  list_price * (1 - discount1 / 100) * (1 - discount2 / 100) = 105 := by
sorry

end shirt_final_price_l2867_286785


namespace ladder_length_proof_l2867_286734

/-- The length of a ladder leaning against a wall. -/
def ladder_length : ℝ := 18.027756377319946

/-- The initial distance of the ladder's bottom from the wall. -/
def initial_bottom_distance : ℝ := 6

/-- The distance the ladder's bottom moves when the top slips. -/
def bottom_slip_distance : ℝ := 12.480564970698127

/-- The distance the ladder's top slips down the wall. -/
def top_slip_distance : ℝ := 4

/-- Theorem stating the length of the ladder given the conditions. -/
theorem ladder_length_proof : 
  ∃ (initial_height : ℝ),
    ladder_length ^ 2 = initial_height ^ 2 + initial_bottom_distance ^ 2 ∧
    ladder_length ^ 2 = (initial_height - top_slip_distance) ^ 2 + bottom_slip_distance ^ 2 :=
by sorry

end ladder_length_proof_l2867_286734


namespace sum_of_tangent_slopes_l2867_286737

/-- The circle with center (2, -1) and radius √2 -/
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 + 1)^2 = 2}

/-- A line passing through the origin with slope k -/
def tangentLine (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = k * p.1}

/-- The set of slopes of lines passing through the origin and tangent to C -/
def tangentSlopes : Set ℝ :=
  {k : ℝ | ∃ p ∈ C, p ∈ tangentLine k ∧ (0, 0) ∈ tangentLine k}

theorem sum_of_tangent_slopes :
  ∃ (k₁ k₂ : ℝ), k₁ ∈ tangentSlopes ∧ k₂ ∈ tangentSlopes ∧ k₁ + k₂ = -2 :=
sorry

end sum_of_tangent_slopes_l2867_286737


namespace no_real_roots_iff_range_m_range_when_necessary_not_sufficient_l2867_286746

-- Define the quadratic equation
def quadratic_equation (a x : ℝ) : Prop :=
  x^2 - 2*a*x + 2*a^2 - a - 6 = 0

-- Define the range of a for no real roots
def no_real_roots (a : ℝ) : Prop :=
  a < -2 ∨ a > 3

-- Define the necessary condition
def necessary_condition (a : ℝ) : Prop :=
  -2 ≤ a ∧ a ≤ 3

-- Define the condition q
def condition_q (m a : ℝ) : Prop :=
  m - 1 ≤ a ∧ a ≤ m + 3

-- Theorem 1: The equation has no real roots iff a is in the specified range
theorem no_real_roots_iff_range (a : ℝ) :
  (∀ x : ℝ, ¬(quadratic_equation a x)) ↔ no_real_roots a :=
sorry

-- Theorem 2: If the necessary condition is true but not sufficient for condition q,
-- then m is in the range [-1, 0]
theorem m_range_when_necessary_not_sufficient :
  (∀ a : ℝ, condition_q m a → necessary_condition a) ∧
  (∃ a : ℝ, necessary_condition a ∧ ¬(condition_q m a)) →
  -1 ≤ m ∧ m ≤ 0 :=
sorry

end no_real_roots_iff_range_m_range_when_necessary_not_sufficient_l2867_286746


namespace vector_perpendicular_l2867_286795

def a : ℝ × ℝ := (2, -1)
def b : ℝ × ℝ := (1, -3)

theorem vector_perpendicular : 
  let diff := (a.1 - b.1, a.2 - b.2)
  a.1 * diff.1 + a.2 * diff.2 = 0 := by sorry

end vector_perpendicular_l2867_286795


namespace sum_of_xy_l2867_286727

theorem sum_of_xy (x y : ℕ) : 
  x > 0 → y > 0 → x < 25 → y < 25 → x + y + x * y = 118 → x + y = 22 := by
  sorry

end sum_of_xy_l2867_286727


namespace power_fraction_simplification_l2867_286770

theorem power_fraction_simplification : (3^2040 + 3^2038) / (3^2040 - 3^2038) = 5/4 := by
  sorry

end power_fraction_simplification_l2867_286770


namespace function_properties_monotonicity_condition_l2867_286709

/-- The function f(x) = ax³ + bx² -/
def f (a b x : ℝ) : ℝ := a * x^3 + b * x^2

/-- The derivative of f(x) -/
def f_derivative (a b x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x

theorem function_properties (a b : ℝ) :
  (f a b 1 = 4) ∧ 
  (f_derivative a b 1 * 1 = -9) →
  (a = 1 ∧ b = 3) :=
sorry

theorem monotonicity_condition (m : ℝ) :
  (∀ x ∈ Set.Icc m (m + 1), f_derivative 1 3 x > 0) →
  (m ≥ 0 ∨ m ≤ -3) :=
sorry

end function_properties_monotonicity_condition_l2867_286709


namespace polar_to_cartesian_l2867_286755

theorem polar_to_cartesian (p θ x y : ℝ) :
  (p = 8 * Real.cos θ) ∧ (x = p * Real.cos θ) ∧ (y = p * Real.sin θ) →
  x^2 + y^2 = 8*x :=
by sorry

end polar_to_cartesian_l2867_286755


namespace child_ticket_cost_l2867_286723

theorem child_ticket_cost (adult_price : ℕ) (num_adults num_children : ℕ) (total_price : ℕ) :
  adult_price = 22 →
  num_adults = 2 →
  num_children = 2 →
  total_price = 58 →
  (total_price - num_adults * adult_price) / num_children = 7 := by
  sorry

end child_ticket_cost_l2867_286723


namespace f_composition_value_l2867_286763

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then 5 * x + 4 else 2^x

def angle_terminal_side_point (α : ℝ) (p : ℝ × ℝ) : Prop :=
  p.2 / p.1 = Real.tan α

theorem f_composition_value (α : ℝ) :
  angle_terminal_side_point α (4, -3) →
  f (f (Real.sin α)) = 2 := by
  sorry

end f_composition_value_l2867_286763


namespace max_green_cards_achievable_green_cards_l2867_286712

/-- Represents the number of cards of each color in the box -/
structure CardCount where
  green : ℕ
  yellow : ℕ

/-- The probability of selecting three cards of the same color -/
def prob_same_color (cc : CardCount) : ℚ :=
  let total := cc.green + cc.yellow
  (cc.green.choose 3 + cc.yellow.choose 3) / total.choose 3

/-- The main theorem stating the maximum number of green cards possible -/
theorem max_green_cards (cc : CardCount) : 
  cc.green + cc.yellow ≤ 2209 →
  prob_same_color cc = 1/3 →
  cc.green ≤ 1092 := by
  sorry

/-- The theorem stating that 1092 green cards is achievable -/
theorem achievable_green_cards : 
  ∃ (cc : CardCount), cc.green + cc.yellow ≤ 2209 ∧ 
  prob_same_color cc = 1/3 ∧ 
  cc.green = 1092 := by
  sorry

end max_green_cards_achievable_green_cards_l2867_286712


namespace function_max_abs_bound_l2867_286710

theorem function_max_abs_bound (a : ℝ) (ha : 0 < a ∧ a < 1) :
  let f (x : ℝ) := a * x^3 + (1 - 4*a) * x^2 + (5*a - 1) * x - 5*a + 3
  let g (x : ℝ) := (1 - a) * x^3 - x^2 + (2 - a) * x - 3*a - 1
  ∀ x : ℝ, max (|f x|) (|g x|) ≥ a + 1 := by
  sorry

end function_max_abs_bound_l2867_286710


namespace inequality_implies_range_l2867_286700

theorem inequality_implies_range (a : ℝ) : 
  (∀ x : ℝ, |2*x + 1| + |x - 2| ≥ a^2 - a + 1/2) → 
  -1 ≤ a ∧ a ≤ 2 := by sorry

end inequality_implies_range_l2867_286700


namespace shopping_money_left_l2867_286726

theorem shopping_money_left (initial_amount : ℝ) (final_amount : ℝ) 
  (spent_percentage : ℝ) (h1 : initial_amount = 4000) 
  (h2 : final_amount = 2800) (h3 : spent_percentage = 0.3) : 
  initial_amount * (1 - spent_percentage) = final_amount := by
  sorry

end shopping_money_left_l2867_286726


namespace teena_current_distance_l2867_286757

/-- Represents the current situation and future state of two drivers on a road -/
structure DrivingSituation where
  teena_speed : ℝ  -- Teena's speed in miles per hour
  poe_speed : ℝ    -- Poe's speed in miles per hour
  time : ℝ          -- Time in hours
  future_distance : ℝ  -- Distance Teena will be ahead of Poe after the given time

/-- Calculates the current distance between two drivers given their future state -/
def current_distance (s : DrivingSituation) : ℝ :=
  ((s.teena_speed - s.poe_speed) * s.time) - s.future_distance

/-- Theorem stating that Teena is currently 7.5 miles behind Poe -/
theorem teena_current_distance :
  let s : DrivingSituation := {
    teena_speed := 55,
    poe_speed := 40,
    time := 1.5,  -- 90 minutes = 1.5 hours
    future_distance := 15
  }
  current_distance s = 7.5 := by
  sorry

end teena_current_distance_l2867_286757


namespace fraction_order_l2867_286768

theorem fraction_order (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (hab : a > b) : 
  (b / a < (b + c) / (a + c)) ∧ 
  ((b + c) / (a + c) < (a + d) / (b + d)) ∧ 
  ((a + d) / (b + d) < a / b) := by
sorry

end fraction_order_l2867_286768


namespace shiela_paintings_l2867_286765

/-- The number of paintings Shiela can give to each grandmother -/
def paintings_per_grandmother : ℕ := 9

/-- The number of grandmothers Shiela has -/
def number_of_grandmothers : ℕ := 2

/-- The total number of paintings Shiela has -/
def total_paintings : ℕ := paintings_per_grandmother * number_of_grandmothers

theorem shiela_paintings : total_paintings = 18 := by
  sorry

end shiela_paintings_l2867_286765


namespace angle_inequality_l2867_286707

theorem angle_inequality : 
  let a : ℝ := (1/2) * Real.cos (6 * π / 180) - (Real.sqrt 3 / 2) * Real.sin (6 * π / 180)
  let b : ℝ := (2 * Real.tan (13 * π / 180)) / (1 - Real.tan (13 * π / 180)^2)
  let c : ℝ := Real.sqrt ((1 - Real.cos (50 * π / 180)) / 2)
  a < c ∧ c < b := by sorry

end angle_inequality_l2867_286707


namespace passengers_at_third_station_l2867_286720

/-- Calculates the number of passengers at the third station given the initial number of passengers and the changes at each station. -/
def passengersAtThirdStation (initialPassengers : ℕ) : ℕ :=
  let afterFirstDrop := initialPassengers - initialPassengers / 3
  let afterFirstAdd := afterFirstDrop + 280
  let afterSecondDrop := afterFirstAdd - afterFirstAdd / 2
  afterSecondDrop + 12

/-- Theorem stating that given 270 initial passengers, the number of passengers at the third station is 242. -/
theorem passengers_at_third_station :
  passengersAtThirdStation 270 = 242 := by
  sorry

#eval passengersAtThirdStation 270

end passengers_at_third_station_l2867_286720


namespace profit_percentage_theorem_l2867_286730

theorem profit_percentage_theorem (selling_price purchase_price : ℝ) 
  (h1 : selling_price > 0) 
  (h2 : purchase_price > 0) 
  (h3 : selling_price > purchase_price) :
  let original_profit_percentage := (selling_price - purchase_price) / purchase_price * 100
  let new_purchase_price := purchase_price * 0.95
  let new_profit_percentage := (selling_price - new_purchase_price) / new_purchase_price * 100
  (new_profit_percentage - original_profit_percentage = 15) → 
  (original_profit_percentage = 185) := by
sorry

end profit_percentage_theorem_l2867_286730


namespace max_value_of_function_l2867_286742

theorem max_value_of_function (x : ℝ) (h : x > 0) : 2 - 9*x - 4/x ≤ -10 := by
  sorry

end max_value_of_function_l2867_286742


namespace adult_tickets_sold_l2867_286784

theorem adult_tickets_sold (adult_price student_price total_tickets total_revenue : ℕ) 
  (h1 : adult_price = 6)
  (h2 : student_price = 3)
  (h3 : total_tickets = 846)
  (h4 : total_revenue = 3846) :
  ∃ (adult_tickets : ℕ), 
    adult_tickets * adult_price + (total_tickets - adult_tickets) * student_price = total_revenue ∧
    adult_tickets = 436 := by
  sorry

end adult_tickets_sold_l2867_286784


namespace f_monotonicity_and_b_range_l2867_286758

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a / (a^2 - 1)) * (a^x - a^(-x))

def g (b : ℝ) (x : ℝ) : ℝ := x^2 + (2*b + 1)*x - b - 1

def prop_p (a b : ℝ) : Prop := ∀ x ∈ Set.Icc (-1) 1, f a x ≤ 2*b

def prop_q (b : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), x₁ ∈ Set.Ioo (-3) (-2) ∧ x₂ ∈ Set.Ioo 0 1 ∧
  g b x₁ = 0 ∧ g b x₂ = 0

theorem f_monotonicity_and_b_range (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f a x₁ < f a x₂) ∧
  {b : ℝ | (prop_p a b ∨ prop_q b) ∧ ¬(prop_p a b ∧ prop_q b)} =
  Set.Ioo (1/5) (1/2) ∪ Set.Ici (5/7) :=
sorry

end f_monotonicity_and_b_range_l2867_286758


namespace stratified_sampling_young_representatives_l2867_286733

/-- Represents the number of young representatives to be selected in a stratified sampling scenario. -/
def young_representatives (total_population : ℕ) (young_population : ℕ) (total_representatives : ℕ) : ℕ :=
  (young_population * total_representatives) / total_population

/-- Theorem stating that for the given population numbers and sampling size, 
    the number of young representatives to be selected is 7. -/
theorem stratified_sampling_young_representatives :
  young_representatives 1000 350 20 = 7 := by
  sorry

end stratified_sampling_young_representatives_l2867_286733


namespace sum_of_factorials_perfect_square_l2867_286704

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sumOfFactorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

def isPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem sum_of_factorials_perfect_square (n : ℕ) :
  isPerfectSquare (sumOfFactorials n) ↔ n = 1 ∨ n = 3 := by
  sorry

end sum_of_factorials_perfect_square_l2867_286704


namespace sqrt_calculation_l2867_286731

theorem sqrt_calculation : (Real.sqrt 8 + Real.sqrt 3) * Real.sqrt 6 - 3 * Real.sqrt 2 = 4 * Real.sqrt 3 := by
  sorry

end sqrt_calculation_l2867_286731


namespace larger_cuboid_height_l2867_286735

/-- Represents the dimensions of a cuboid -/
structure CuboidDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a cuboid given its dimensions -/
def cuboidVolume (d : CuboidDimensions) : ℝ :=
  d.length * d.width * d.height

theorem larger_cuboid_height :
  let smallerCuboid : CuboidDimensions := ⟨5, 6, 3⟩
  let largerCuboidBase : CuboidDimensions := ⟨18, 15, 2⟩
  let numSmallerCuboids : ℕ := 6
  cuboidVolume largerCuboidBase = numSmallerCuboids * cuboidVolume smallerCuboid := by
  sorry

end larger_cuboid_height_l2867_286735


namespace set_operations_l2867_286766

def U : Set ℤ := {1, 2, 3, 4, 5, 6, 7, 8}

def A : Set ℤ := {x | x^2 - 3*x + 2 = 0}

def B : Set ℤ := {x | 1 ≤ x ∧ x ≤ 5}

def C : Set ℤ := {x | 2 < x ∧ x < 9}

theorem set_operations :
  (A ∪ (B ∩ C) = {1, 2, 3, 4, 5}) ∧
  ((U.compl ∩ B) ∪ (U.compl ∩ C) = {1, 2, 6, 7, 8}) := by
  sorry

end set_operations_l2867_286766


namespace tshirt_cost_l2867_286715

theorem tshirt_cost (original_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) :
  original_price = 240 ∧ 
  discount_rate = 0.2 ∧ 
  profit_rate = 0.2 →
  ∃ (cost : ℝ), cost = 160 ∧ 
    cost * (1 + profit_rate) = original_price * (1 - discount_rate) :=
by sorry

end tshirt_cost_l2867_286715


namespace power_of_prime_iff_only_prime_factor_l2867_286780

theorem power_of_prime_iff_only_prime_factor (p n : ℕ) : 
  Prime p → (∃ k : ℕ, n = p ^ k) ↔ (∀ q : ℕ, Prime q → q ∣ n → q = p) :=
sorry

end power_of_prime_iff_only_prime_factor_l2867_286780


namespace least_coins_seventeen_coins_least_possible_coins_l2867_286743

theorem least_coins (a : ℕ) : (a % 7 = 3 ∧ a % 4 = 1) → a ≥ 17 := by
  sorry

theorem seventeen_coins : 17 % 7 = 3 ∧ 17 % 4 = 1 := by
  sorry

theorem least_possible_coins : ∃ (a : ℕ), a % 7 = 3 ∧ a % 4 = 1 ∧ ∀ (b : ℕ), (b % 7 = 3 ∧ b % 4 = 1) → a ≤ b := by
  sorry

end least_coins_seventeen_coins_least_possible_coins_l2867_286743


namespace fraction_sum_l2867_286773

theorem fraction_sum : (1 : ℚ) / 6 + (2 : ℚ) / 9 + (1 : ℚ) / 3 = (13 : ℚ) / 18 := by
  sorry

end fraction_sum_l2867_286773


namespace line_through_points_l2867_286724

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line defined by two points -/
structure Line where
  p1 : Point
  p2 : Point

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem line_through_points : 
  let p1 : Point := ⟨2, 5⟩
  let p2 : Point := ⟨6, 17⟩
  let p3 : Point := ⟨10, 29⟩
  let p4 : Point := ⟨34, s⟩
  collinear p1 p2 p3 ∧ collinear p1 p2 p4 → s = 101 := by
  sorry


end line_through_points_l2867_286724


namespace functional_equation_solution_l2867_286793

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x + y) = f (x - y) + 4 * x * y) →
  ∃ c : ℝ, ∀ x : ℝ, f x = x^2 + c :=
by sorry

end functional_equation_solution_l2867_286793


namespace total_rectangles_is_176_l2867_286719

/-- The number of gray cells in the picture -/
def total_gray_cells : ℕ := 40

/-- The number of "blue" cells (a subset of gray cells) -/
def blue_cells : ℕ := 36

/-- The number of "red" cells (the remaining subset of gray cells) -/
def red_cells : ℕ := total_gray_cells - blue_cells

/-- The number of unique rectangles containing each blue cell -/
def rectangles_per_blue_cell : ℕ := 4

/-- The number of unique rectangles containing each red cell -/
def rectangles_per_red_cell : ℕ := 8

/-- The total number of checkered rectangles containing exactly one gray cell -/
def total_rectangles : ℕ := blue_cells * rectangles_per_blue_cell + red_cells * rectangles_per_red_cell

theorem total_rectangles_is_176 : total_rectangles = 176 := by
  sorry

end total_rectangles_is_176_l2867_286719


namespace duplicate_page_number_l2867_286703

theorem duplicate_page_number (n : ℕ) (p : ℕ) : 
  (n ≥ 70) →
  (n ≤ 71) →
  (n * (n + 1)) / 2 + p = 2550 →
  p = 80 := by
sorry

end duplicate_page_number_l2867_286703


namespace train_length_l2867_286714

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 100 → time_s = 9 → 
  ∃ (length_m : ℝ), abs (length_m - 250.02) < 0.01 :=
by sorry

end train_length_l2867_286714


namespace parallel_lines_imply_a_equals_one_l2867_286745

/-- Two lines are parallel if and only if their slopes are equal -/
def parallel_lines (m1 m2 : ℝ) : Prop := m1 = m2

/-- The first line: y = ax - 2 -/
def line1 (a x : ℝ) : ℝ := a * x - 2

/-- The second line: y = (2-a)x + 1 -/
def line2 (a x : ℝ) : ℝ := (2 - a) * x + 1

theorem parallel_lines_imply_a_equals_one (a : ℝ) :
  parallel_lines a (2 - a) → a = 1 := by
  sorry

end parallel_lines_imply_a_equals_one_l2867_286745


namespace fractional_equation_solution_range_l2867_286771

theorem fractional_equation_solution_range (x m : ℝ) : 
  ((2 * x - m) / (x + 1) = 3) → 
  (x < 0) → 
  (m > -3 ∧ m ≠ -2) :=
by sorry

end fractional_equation_solution_range_l2867_286771


namespace xiao_li_commute_l2867_286729

/-- Xiao Li's commute problem -/
theorem xiao_li_commute
  (distance : ℝ)
  (walk_late : ℝ)
  (bike_early : ℝ)
  (bike_speed_ratio : ℝ)
  (bike_breakdown_distance : ℝ)
  (h_distance : distance = 4.5)
  (h_walk_late : walk_late = 5 / 60)
  (h_bike_early : bike_early = 10 / 60)
  (h_bike_speed_ratio : bike_speed_ratio = 1.5)
  (h_bike_breakdown_distance : bike_breakdown_distance = 1.5) :
  ∃ (walk_speed bike_speed min_run_speed : ℝ),
    walk_speed = 6 ∧
    bike_speed = 9 ∧
    min_run_speed = 7.2 ∧
    distance / walk_speed - walk_late = distance / bike_speed + bike_early ∧
    bike_speed = bike_speed_ratio * walk_speed ∧
    bike_breakdown_distance / bike_speed +
      (distance - bike_breakdown_distance) / min_run_speed ≤
        distance / bike_speed + bike_early - 5 / 60 :=
by sorry

end xiao_li_commute_l2867_286729


namespace defective_pens_probability_l2867_286718

theorem defective_pens_probability (total_pens : Nat) (defective_pens : Nat) (bought_pens : Nat) :
  total_pens = 10 →
  defective_pens = 2 →
  bought_pens = 2 →
  (((total_pens - defective_pens : ℚ) / total_pens) * 
   ((total_pens - defective_pens - 1 : ℚ) / (total_pens - 1))) = 0.6222222222222222 := by
  sorry

end defective_pens_probability_l2867_286718


namespace incorrect_translation_l2867_286750

/-- Represents a parabola of the form y = (x + a)² + b -/
structure Parabola where
  a : ℝ
  b : ℝ

/-- Checks if a parabola passes through the origin -/
def passes_through_origin (p : Parabola) : Prop :=
  0 = (0 + p.a)^2 + p.b

/-- Translates a parabola vertically -/
def translate_vertical (p : Parabola) (d : ℝ) : Parabola :=
  { a := p.a, b := p.b - d }

theorem incorrect_translation :
  let original := Parabola.mk 3 (-4)
  let translated := translate_vertical original 4
  ¬ passes_through_origin translated :=
by sorry

end incorrect_translation_l2867_286750


namespace bus_ride_distance_l2867_286782

theorem bus_ride_distance 
  (total_time : ℝ) 
  (bus_speed : ℝ) 
  (walking_speed : ℝ) 
  (h1 : total_time = 8) 
  (h2 : bus_speed = 9) 
  (h3 : walking_speed = 3) : 
  ∃ d : ℝ, d = 18 ∧ d / bus_speed + d / walking_speed = total_time :=
by
  sorry

end bus_ride_distance_l2867_286782


namespace tower_of_hanoi_minimum_moves_five_disks_minimum_moves_l2867_286738

/-- Minimum number of moves required to solve the Tower of Hanoi puzzle with n disks -/
def tower_of_hanoi_moves (n : ℕ) : ℕ := 2^n - 1

/-- The number of disks in our specific problem -/
def num_disks : ℕ := 5

theorem tower_of_hanoi_minimum_moves :
  ∀ n : ℕ, tower_of_hanoi_moves n = 2^n - 1 :=
sorry

theorem five_disks_minimum_moves :
  tower_of_hanoi_moves num_disks = 31 :=
sorry

end tower_of_hanoi_minimum_moves_five_disks_minimum_moves_l2867_286738


namespace isosceles_triangle_perimeter_l2867_286797

/-- An isosceles triangle with two sides of lengths 3 and 7 has a perimeter of 17 -/
theorem isosceles_triangle_perimeter : ∀ a b c : ℝ,
  a > 0 → b > 0 → c > 0 →
  (a = 3 ∧ b = 7 ∧ c = 7) ∨ (a = 7 ∧ b = 3 ∧ c = 7) →
  a + b + c = 17 := by
  sorry

end isosceles_triangle_perimeter_l2867_286797


namespace inheritance_tax_calculation_l2867_286756

theorem inheritance_tax_calculation (x : ℝ) : 
  (0.2 * x + 0.1 * (x - 0.2 * x) = 10500) → x = 37500 := by
  sorry

end inheritance_tax_calculation_l2867_286756


namespace two_bags_below_threshold_probability_l2867_286732

-- Define the normal distribution parameters
def μ : ℝ := 500
def σ : ℝ := 5

-- Define the threshold weight
def threshold : ℝ := 485

-- Define the probability of selecting one bag below the threshold
def prob_one_bag : ℝ := 0.0013

-- Theorem statement
theorem two_bags_below_threshold_probability :
  let prob_two_bags := prob_one_bag * prob_one_bag
  prob_two_bags < 2e-6 := by sorry

end two_bags_below_threshold_probability_l2867_286732


namespace sqrt_equation_solution_l2867_286759

theorem sqrt_equation_solution (w : ℝ) :
  (Real.sqrt 1.1 / Real.sqrt 0.81 + Real.sqrt 1.44 / Real.sqrt w = 2.879628878919216) →
  w = 0.49 := by
  sorry

end sqrt_equation_solution_l2867_286759


namespace strawberry_juice_problem_l2867_286754

theorem strawberry_juice_problem (T : ℚ) 
  (h1 : T > 0)
  (h2 : (5/6 * T - 2/5 * (5/6 * T) - 2/3 * (5/6 * T - 2/5 * (5/6 * T))) = 120) : 
  T = 720 := by
  sorry

end strawberry_juice_problem_l2867_286754


namespace problem_statement_l2867_286772

theorem problem_statement (a b c : ℝ) (h : (2:ℝ)^a = (3:ℝ)^b ∧ (3:ℝ)^b = (18:ℝ)^c ∧ (18:ℝ)^c < 1) :
  b < 2*c ∧ (a + b)/c > 3 + 2*Real.sqrt 2 := by
  sorry

end problem_statement_l2867_286772


namespace slope_of_MN_constant_sum_of_reciprocals_l2867_286775

/- Ellipse C₁ -/
def C₁ (b : ℝ) (x y : ℝ) : Prop := x^2/8 + y^2/b^2 = 1 ∧ b > 0

/- Parabola C₂ -/
def C₂ (x y : ℝ) : Prop := y^2 = 8*x

/- Right focus F₂ -/
def F₂ : ℝ × ℝ := (2, 0)

/- Theorem for the slope of line MN -/
theorem slope_of_MN (b : ℝ) (M N : ℝ × ℝ) :
  C₁ b M.1 M.2 → C₁ b N.1 N.2 → ((M.1 + N.1)/2, (M.2 + N.2)/2) = (1, 1) →
  (N.2 - M.2) / (N.1 - M.1) = -1/2 :=
sorry

/- Theorem for the constant sum of reciprocals -/
theorem constant_sum_of_reciprocals (b : ℝ) (A B C D : ℝ × ℝ) (m n : ℝ) :
  C₁ b A.1 A.2 → C₁ b B.1 B.2 → C₁ b C.1 C.2 → C₁ b D.1 D.2 →
  ((A.1 - F₂.1) * (B.1 - F₂.1) + (A.2 - F₂.2) * (B.2 - F₂.2) = 0) →
  ((C.1 - F₂.1) * (D.1 - F₂.1) + (C.2 - F₂.2) * (D.2 - F₂.2) = 0) →
  m = Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) →
  n = Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) →
  1/m + 1/n = 3 * Real.sqrt 2 / 8 :=
sorry

end slope_of_MN_constant_sum_of_reciprocals_l2867_286775


namespace juvenile_female_percentage_l2867_286796

/-- Represents the population of alligators on Lagoon Island -/
structure AlligatorPopulation where
  total : ℕ
  males : ℕ
  adult_females : ℕ
  juvenile_females : ℕ

/-- Conditions for the Lagoon Island alligator population -/
def lagoon_conditions (pop : AlligatorPopulation) : Prop :=
  pop.males = pop.total / 2 ∧
  pop.males = 25 ∧
  pop.adult_females = 15 ∧
  pop.juvenile_females = pop.total / 2 - pop.adult_females

/-- Theorem: The percentage of juvenile female alligators is 40% -/
theorem juvenile_female_percentage (pop : AlligatorPopulation) 
  (h : lagoon_conditions pop) : 
  (pop.juvenile_females : ℚ) / (pop.total / 2 : ℚ) = 2/5 := by
  sorry

#check juvenile_female_percentage

end juvenile_female_percentage_l2867_286796


namespace students_walking_home_l2867_286740

theorem students_walking_home (bus_fraction car_fraction scooter_fraction : ℚ)
  (h1 : bus_fraction = 2/5)
  (h2 : car_fraction = 1/5)
  (h3 : scooter_fraction = 1/8)
  (h4 : bus_fraction + car_fraction + scooter_fraction < 1) :
  1 - (bus_fraction + car_fraction + scooter_fraction) = 11/40 := by
sorry

end students_walking_home_l2867_286740
