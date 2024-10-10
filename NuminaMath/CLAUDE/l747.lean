import Mathlib

namespace count_polynomials_l747_74786

/-- A function to determine if an expression is a polynomial -/
def is_polynomial (expr : String) : Bool :=
  match expr with
  | "x^2+2" => true
  | "1/a+4" => false
  | "3ab^2/7" => true
  | "ab/c" => false
  | "-5x" => true
  | "0" => true
  | _ => false

/-- The list of expressions to check -/
def expressions : List String := ["x^2+2", "1/a+4", "3ab^2/7", "ab/c", "-5x", "0"]

/-- Theorem stating that there are exactly 4 polynomial expressions in the given list -/
theorem count_polynomials : 
  (expressions.filter is_polynomial).length = 4 := by sorry

end count_polynomials_l747_74786


namespace quadratic_unique_root_l747_74768

/-- A function that represents the quadratic equation (m-4)x^2 - 2mx - m - 6 -/
def f (m : ℝ) (x : ℝ) : ℝ := (m - 4) * x^2 - 2 * m * x - m - 6

/-- Condition for the quadratic function to have exactly one root -/
def has_unique_root (m : ℝ) : Prop :=
  (∃ x : ℝ, f m x = 0) ∧ (∀ x y : ℝ, f m x = 0 → f m y = 0 → x = y)

theorem quadratic_unique_root (m : ℝ) :
  has_unique_root m → m = -4 ∨ m = 3 ∨ m = 4 := by sorry

end quadratic_unique_root_l747_74768


namespace total_plants_l747_74701

def garden_problem (basil oregano thyme rosemary : ℕ) : Prop :=
  oregano = 2 * basil + 2 ∧
  thyme = 3 * basil - 3 ∧
  rosemary = (basil + thyme) / 2 ∧
  basil = 5 ∧
  basil + oregano + thyme + rosemary ≤ 50

theorem total_plants (basil oregano thyme rosemary : ℕ) :
  garden_problem basil oregano thyme rosemary →
  basil + oregano + thyme + rosemary = 37 :=
by
  sorry

end total_plants_l747_74701


namespace distance_calculation_l747_74740

/-- The distance between Maxwell's and Brad's homes -/
def distance_between_homes : ℝ := 24

/-- Maxwell's walking speed in km/h -/
def maxwell_speed : ℝ := 4

/-- Brad's running speed in km/h -/
def brad_speed : ℝ := 6

/-- Time difference between Maxwell's and Brad's start times in hours -/
def time_difference : ℝ := 1

/-- Total time Maxwell walks before meeting Brad in hours -/
def total_time : ℝ := 3

theorem distance_calculation :
  distance_between_homes = maxwell_speed * total_time + brad_speed * (total_time - time_difference) :=
by sorry

end distance_calculation_l747_74740


namespace divisibility_by_37_l747_74794

def N (x y : ℕ) : ℕ := 300070003 + 1000000 * x + 100 * y

theorem divisibility_by_37 :
  ∀ x y : ℕ, x ≤ 9 ∧ y ≤ 9 →
  (37 ∣ N x y ↔ (x = 8 ∧ y = 1) ∨ (x = 4 ∧ y = 4) ∨ (x = 0 ∧ y = 7)) :=
by sorry

end divisibility_by_37_l747_74794


namespace sum_of_coordinates_B_l747_74712

/-- Given that M(3,7) is the midpoint of AB and A(9,3), prove that the sum of B's coordinates is 8 -/
theorem sum_of_coordinates_B (A B M : ℝ × ℝ) : 
  A = (9, 3) → M = (3, 7) → M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) → 
  B.1 + B.2 = 8 := by
  sorry

end sum_of_coordinates_B_l747_74712


namespace chosen_number_calculation_l747_74753

theorem chosen_number_calculation (x : ℕ) (h : x = 30) : x * 8 - 138 = 102 := by
  sorry

end chosen_number_calculation_l747_74753


namespace complex_magnitude_problem_l747_74709

theorem complex_magnitude_problem (i : ℂ) (z : ℂ) :
  i^2 = -1 →
  z = (1 - i) / (2 + i) →
  Complex.abs z = Real.sqrt 10 / 5 := by
  sorry

end complex_magnitude_problem_l747_74709


namespace a_leq_0_necessary_not_sufficient_l747_74769

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then 2 * x^2 + a * x - 3/2
  else 2 * a * x^2 + x

-- Define what it means for a function to be monotonically decreasing
def MonotonicallyDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≥ f y

-- Theorem statement
theorem a_leq_0_necessary_not_sufficient :
  (∃ a : ℝ, a ≤ 0 ∧ ¬(MonotonicallyDecreasing (f a))) ∧
  (∀ a : ℝ, MonotonicallyDecreasing (f a) → a ≤ 0) :=
sorry

end a_leq_0_necessary_not_sufficient_l747_74769


namespace cookie_jar_theorem_l747_74726

def cookie_jar_problem (initial_amount doris_spent : ℕ) : Prop :=
  let martha_spent := doris_spent / 2
  let total_spent := doris_spent + martha_spent
  let remaining_amount := initial_amount - total_spent
  remaining_amount = 15

theorem cookie_jar_theorem :
  cookie_jar_problem 24 6 := by
  sorry

end cookie_jar_theorem_l747_74726


namespace expression_factorization_l747_74792

theorem expression_factorization (b : ℝ) :
  (8 * b^4 - 100 * b^3 + 14 * b^2) - (3 * b^4 - 10 * b^3 + 14 * b^2) = 5 * b^3 * (b - 18) := by
  sorry

end expression_factorization_l747_74792


namespace convention_handshakes_eq_680_l747_74749

/-- Represents the number of handshakes at a twins and quadruplets convention --/
def convention_handshakes : ℕ := by
  -- Define the number of twin sets and quadruplet sets
  let twin_sets : ℕ := 8
  let quad_sets : ℕ := 5

  -- Calculate total number of twins and quadruplets
  let total_twins : ℕ := twin_sets * 2
  let total_quads : ℕ := quad_sets * 4

  -- Calculate handshakes among twins
  let twin_handshakes : ℕ := (total_twins * (total_twins - 2)) / 2

  -- Calculate handshakes among quadruplets
  let quad_handshakes : ℕ := (total_quads * (total_quads - 4)) / 2

  -- Calculate cross handshakes between twins and quadruplets
  let cross_handshakes : ℕ := total_twins * (2 * total_quads / 3)

  -- Sum all handshakes
  exact twin_handshakes + quad_handshakes + cross_handshakes

/-- Theorem stating that the total number of handshakes is 680 --/
theorem convention_handshakes_eq_680 : convention_handshakes = 680 := by
  sorry

end convention_handshakes_eq_680_l747_74749


namespace quadratic_equation_distinct_roots_l747_74744

theorem quadratic_equation_distinct_roots :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ -x₁^2 - 3*x₁ + 3 = 0 ∧ -x₂^2 - 3*x₂ + 3 = 0 := by
  sorry

end quadratic_equation_distinct_roots_l747_74744


namespace real_number_inequality_l747_74747

theorem real_number_inequality (x : Fin 8 → ℝ) (h : ∀ i j, i ≠ j → x i ≠ x j) :
  ∃ i j, i ≠ j ∧ 0 < (x i - x j) / (1 + x i * x j) ∧ (x i - x j) / (1 + x i * x j) < Real.tan (π / 7) := by
  sorry

end real_number_inequality_l747_74747


namespace subset_gcd_property_l747_74784

theorem subset_gcd_property (A : Finset ℕ) 
  (h1 : A ⊆ Finset.range 2007)
  (h2 : A.card = 1004) :
  (∃ a b c : ℕ, a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ (Nat.gcd a b ∣ c)) ∧
  (∃ a b c : ℕ, a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ ¬(Nat.gcd a b ∣ c)) := by
sorry

end subset_gcd_property_l747_74784


namespace matrix_inverse_proof_l747_74721

def A : Matrix (Fin 2) (Fin 2) ℚ := !![4, 5; -2, 9]

def A_inv : Matrix (Fin 2) (Fin 2) ℚ := !![9/46, -5/46; 1/23, 2/23]

theorem matrix_inverse_proof :
  A * A_inv = 1 ∧ A_inv * A = 1 := by
  sorry

end matrix_inverse_proof_l747_74721


namespace meadow_area_is_24_l747_74774

/-- The area of a meadow that was mowed in two days -/
def meadow_area : ℝ → Prop :=
  fun x => 
    -- Day 1: Half of the meadow plus 3 hectares
    let day1 := x / 2 + 3
    -- Remaining area after day 1
    let remaining := x - day1
    -- Day 2: One-third of the remaining area plus 6 hectares
    let day2 := remaining / 3 + 6
    -- The entire meadow is mowed after two days
    day1 + day2 = x

/-- Theorem: The area of the meadow is 24 hectares -/
theorem meadow_area_is_24 : meadow_area 24 := by
  sorry

#check meadow_area_is_24

end meadow_area_is_24_l747_74774


namespace cone_volume_l747_74788

/-- Given a cone with base radius 3 and lateral surface area 15π, its volume is 12π. -/
theorem cone_volume (r h : ℝ) : 
  r = 3 → 
  π * r * (r^2 + h^2).sqrt = 15 * π → 
  (1/3) * π * r^2 * h = 12 * π := by
  sorry

end cone_volume_l747_74788


namespace lcm_of_1428_and_924_l747_74799

theorem lcm_of_1428_and_924 : Nat.lcm 1428 924 = 15708 := by
  sorry

end lcm_of_1428_and_924_l747_74799


namespace isosceles_triangle_l747_74797

theorem isosceles_triangle (a b c : ℝ) (α β γ : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < α ∧ 0 < β ∧ 0 < γ →
  α + β + γ = π →
  a + b = Real.tan (γ / 2) * (a * Real.tan α + b * Real.tan β) →
  α = β := by
  sorry

end isosceles_triangle_l747_74797


namespace sqrt_sum_geq_product_sum_l747_74748

theorem sqrt_sum_geq_product_sum {x y z : ℝ} (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 3) : Real.sqrt x + Real.sqrt y + Real.sqrt z ≥ x * y + y * z + z * x := by
  sorry

end sqrt_sum_geq_product_sum_l747_74748


namespace sum_of_digits_of_sum_of_digits_of_1962_digit_number_div_by_9_l747_74715

-- Define a 1962-digit number
def is_1962_digit_number (n : ℕ) : Prop :=
  10^1961 ≤ n ∧ n < 10^1962

-- Define the sum of digits function
def sum_of_digits (n : ℕ) : ℕ :=
  sorry

-- Define the property of being divisible by 9
def divisible_by_9 (n : ℕ) : Prop :=
  n % 9 = 0

-- State the theorem
theorem sum_of_digits_of_sum_of_digits_of_1962_digit_number_div_by_9 
  (n : ℕ) (a b c : ℕ) : 
  is_1962_digit_number n → 
  divisible_by_9 n → 
  a = sum_of_digits n → 
  b = sum_of_digits a → 
  c = sum_of_digits b → 
  c = 9 :=
sorry

end sum_of_digits_of_sum_of_digits_of_1962_digit_number_div_by_9_l747_74715


namespace regular_polygon_sides_l747_74743

theorem regular_polygon_sides (n : ℕ) (interior_angle : ℝ) (exterior_angle : ℝ) : 
  interior_angle = 150 →
  exterior_angle = 180 - interior_angle →
  n * exterior_angle = 360 →
  n = 12 := by
  sorry

#check regular_polygon_sides

end regular_polygon_sides_l747_74743


namespace least_trees_required_l747_74716

theorem least_trees_required (n : ℕ) : 
  (n > 0 ∧ 4 ∣ n ∧ 5 ∣ n ∧ 6 ∣ n) → 
  (∀ m : ℕ, m > 0 ∧ 4 ∣ m ∧ 5 ∣ m ∧ 6 ∣ m → n ≤ m) → 
  n = 60 := by
sorry

end least_trees_required_l747_74716


namespace twentieth_digit_of_half_power_twenty_l747_74720

theorem twentieth_digit_of_half_power_twenty (n : ℕ) : n = 20 → 
  ∃ (x : ℚ), x = (1/2)^20 ∧ 
  (∃ (a b : ℕ), x = a / (10^n) ∧ x < (a + 1) / (10^n) ∧ a % 10 = 1) :=
sorry

end twentieth_digit_of_half_power_twenty_l747_74720


namespace ball_probability_l747_74736

theorem ball_probability (p_red_yellow p_red_white : ℝ) 
  (h1 : p_red_yellow = 0.4) 
  (h2 : p_red_white = 0.9) : 
  ∃ (p_red p_yellow p_white : ℝ), 
    p_red + p_yellow = p_red_yellow ∧ 
    p_red + p_white = p_red_white ∧ 
    p_yellow + p_white = 0.7 := by
  sorry

end ball_probability_l747_74736


namespace sum_smallest_largest_primes_1_to_50_l747_74756

theorem sum_smallest_largest_primes_1_to_50 :
  (∃ p q : ℕ, 
    Prime p ∧ Prime q ∧
    1 < p ∧ p ≤ 50 ∧
    1 < q ∧ q ≤ 50 ∧
    (∀ r : ℕ, Prime r ∧ 1 < r ∧ r ≤ 50 → p ≤ r ∧ r ≤ q) ∧
    p + q = 49) :=
by sorry

end sum_smallest_largest_primes_1_to_50_l747_74756


namespace family_probability_l747_74782

theorem family_probability (p_boy p_girl : ℝ) (h1 : p_boy = 1 / 2) (h2 : p_girl = 1 / 2) :
  let p_at_least_one_each := 1 - (p_boy ^ 4 + p_girl ^ 4)
  p_at_least_one_each = 7 / 8 := by
sorry

end family_probability_l747_74782


namespace liar_identification_l747_74775

def original_number : ℕ := 2014315

def swap_digits (n : ℕ) (i j : ℕ) : ℕ := sorry

def is_divisible_by (n m : ℕ) : Prop := n % m = 0

def statement_A (cards : Finset ℕ) : Prop :=
  ∃ (i j : ℕ), i ∈ cards ∧ j ∈ cards ∧ i ≠ j ∧
  is_divisible_by (swap_digits original_number i j) 8

def statement_B (cards : Finset ℕ) : Prop :=
  ∀ (i j : ℕ), i ∈ cards → j ∈ cards → i ≠ j →
  ¬is_divisible_by (swap_digits original_number i j) 9

def statement_C (cards : Finset ℕ) : Prop :=
  ∃ (i j : ℕ), i ∈ cards ∧ j ∈ cards ∧ i ≠ j ∧
  is_divisible_by (swap_digits original_number i j) 10

def statement_D (cards : Finset ℕ) : Prop :=
  ∃ (i j : ℕ), i ∈ cards ∧ j ∈ cards ∧ i ≠ j ∧
  is_divisible_by (swap_digits original_number i j) 11

theorem liar_identification :
  ∃ (cards_A cards_B cards_C cards_D : Finset ℕ),
    cards_A.card ≤ 2 ∧ cards_B.card ≤ 2 ∧ cards_C.card ≤ 2 ∧ cards_D.card ≤ 2 ∧
    cards_A ∪ cards_B ∪ cards_C ∪ cards_D = {0, 1, 2, 3, 4, 5} ∧
    cards_A ∩ cards_B = ∅ ∧ cards_A ∩ cards_C = ∅ ∧ cards_A ∩ cards_D = ∅ ∧
    cards_B ∩ cards_C = ∅ ∧ cards_B ∩ cards_D = ∅ ∧ cards_C ∩ cards_D = ∅ ∧
    statement_A cards_A ∧ statement_B cards_B ∧ ¬statement_C cards_C ∧ statement_D cards_D :=
by sorry

end liar_identification_l747_74775


namespace incorrect_multiplication_l747_74762

theorem incorrect_multiplication : (79133 * 111107) % 9 ≠ 8792240231 % 9 := by
  sorry

end incorrect_multiplication_l747_74762


namespace tarantulas_needed_l747_74783

/-- The number of legs for each animal type --/
def legs_per_chimp : ℕ := 4
def legs_per_lion : ℕ := 4
def legs_per_lizard : ℕ := 4
def legs_per_tarantula : ℕ := 8

/-- The number of animals already seen --/
def chimps_seen : ℕ := 12
def lions_seen : ℕ := 8
def lizards_seen : ℕ := 5

/-- The total number of legs Borgnine wants to see --/
def total_legs_goal : ℕ := 1100

/-- Theorem: The number of tarantulas needed to reach the total legs goal --/
theorem tarantulas_needed : 
  (chimps_seen * legs_per_chimp + 
   lions_seen * legs_per_lion + 
   lizards_seen * legs_per_lizard + 
   125 * legs_per_tarantula) = total_legs_goal :=
by sorry

end tarantulas_needed_l747_74783


namespace graphic_artist_pages_sum_l747_74790

theorem graphic_artist_pages_sum (n : ℕ) (a₁ d : ℝ) : 
  n = 15 ∧ a₁ = 3 ∧ d = 2 → 
  (n / 2 : ℝ) * (2 * a₁ + (n - 1) * d) = 255 := by
  sorry

end graphic_artist_pages_sum_l747_74790


namespace fraction_sum_l747_74765

theorem fraction_sum (m n : ℕ) (hcoprime : Nat.Coprime m n) 
  (heq : (2013 * 2013) / (2014 * 2014 + 2012) = n / m) : 
  m + n = 1343 := by
  sorry

end fraction_sum_l747_74765


namespace stock_price_calculation_l747_74755

/-- Calculates the final stock price after two years of changes -/
def final_stock_price (initial_price : ℝ) (first_year_increase : ℝ) (second_year_decrease : ℝ) : ℝ :=
  let price_after_first_year := initial_price * (1 + first_year_increase)
  price_after_first_year * (1 - second_year_decrease)

/-- Theorem stating that given the specific conditions, the final stock price is $151.2 -/
theorem stock_price_calculation :
  final_stock_price 120 0.8 0.3 = 151.2 := by
  sorry

end stock_price_calculation_l747_74755


namespace paper_I_max_mark_l747_74717

/-- The maximum mark for Paper I -/
def max_mark : ℕ := 262

/-- The passing percentage for Paper I -/
def passing_percentage : ℚ := 65 / 100

/-- The marks scored by the candidate -/
def scored_marks : ℕ := 112

/-- The marks by which the candidate failed -/
def failed_by : ℕ := 58

/-- Theorem stating that the maximum mark for Paper I is 262 -/
theorem paper_I_max_mark :
  (↑max_mark * passing_percentage).floor = scored_marks + failed_by :=
sorry

end paper_I_max_mark_l747_74717


namespace sheet_length_is_30_l747_74764

/-- Represents the dimensions and usage of a typist's sheet. -/
structure TypistSheet where
  width : ℝ
  length : ℝ
  sideMargin : ℝ
  topBottomMargin : ℝ
  usagePercentage : ℝ

/-- Calculates the length of a typist's sheet given the specifications. -/
def calculateSheetLength (sheet : TypistSheet) : ℝ :=
  sheet.length

/-- Theorem stating that the length of the sheet is 30 cm under the given conditions. -/
theorem sheet_length_is_30 (sheet : TypistSheet)
    (h1 : sheet.width = 20)
    (h2 : sheet.sideMargin = 2)
    (h3 : sheet.topBottomMargin = 3)
    (h4 : sheet.usagePercentage = 64)
    (h5 : (sheet.width - 2 * sheet.sideMargin) * (sheet.length - 2 * sheet.topBottomMargin) = 
          sheet.usagePercentage / 100 * sheet.width * sheet.length) :
    calculateSheetLength sheet = 30 := by
  sorry

#check sheet_length_is_30

end sheet_length_is_30_l747_74764


namespace class_size_proof_l747_74760

theorem class_size_proof (n : ℕ) : 
  (n / 6 : ℚ) = (n / 18 : ℚ) + 4 →  -- One-sixth wear glasses, split into girls and boys
  n = 36 :=
by
  sorry

#check class_size_proof

end class_size_proof_l747_74760


namespace max_distance_for_given_tires_l747_74731

/-- Represents the maximum distance a car can travel with tire swapping -/
def max_distance (front_tire_life rear_tire_life : ℕ) : ℕ :=
  sorry

/-- Theorem stating the maximum distance for the given tire lifespans -/
theorem max_distance_for_given_tires :
  max_distance 24000 36000 = 28800 := by
  sorry

end max_distance_for_given_tires_l747_74731


namespace nested_f_application_l747_74733

def f (x : ℝ) : ℝ := x + 1

theorem nested_f_application : f (f (f (f (f 3)))) = 8 := by sorry

end nested_f_application_l747_74733


namespace calcium_oxide_weight_l747_74705

-- Define atomic weights
def atomic_weight_Ca : Real := 40.08
def atomic_weight_O : Real := 16.00

-- Define the compound
structure Compound where
  calcium : Nat
  oxygen : Nat

-- Define molecular weight calculation
def molecular_weight (c : Compound) : Real :=
  c.calcium * atomic_weight_Ca + c.oxygen * atomic_weight_O

-- Theorem to prove
theorem calcium_oxide_weight :
  molecular_weight { calcium := 1, oxygen := 1 } = 56.08 := by
  sorry

end calcium_oxide_weight_l747_74705


namespace remainder_theorem_l747_74798

theorem remainder_theorem (x : ℤ) : x % 63 = 25 → x % 8 = 1 := by
  sorry

end remainder_theorem_l747_74798


namespace g_of_negative_four_l747_74738

-- Define the function g
def g (x : ℝ) : ℝ := 5 * x + 2

-- State the theorem
theorem g_of_negative_four : g (-4) = -18 := by
  sorry

end g_of_negative_four_l747_74738


namespace lighthouse_angle_elevation_l747_74766

/-- Given a lighthouse and two ships, proves that the angle of elevation from one ship is 30 degrees -/
theorem lighthouse_angle_elevation 
  (h : ℝ) -- height of the lighthouse
  (d : ℝ) -- distance between the ships
  (θ₁ : ℝ) -- angle of elevation from the first ship
  (θ₂ : ℝ) -- angle of elevation from the second ship
  (h_height : h = 100) -- lighthouse height is 100 m
  (h_distance : d = 273.2050807568877) -- distance between ships
  (h_angle₂ : θ₂ = 45 * π / 180) -- angle from second ship is 45°
  : θ₁ = 30 * π / 180 := by 
sorry


end lighthouse_angle_elevation_l747_74766


namespace problem_solution_l747_74795

theorem problem_solution (x y z : ℝ) (hx : x = 7) (hy : y = -2) (hz : z = 4) :
  ((x - 2*y)^y) / z = 1 / 484 := by
  sorry

end problem_solution_l747_74795


namespace nonparallel_side_length_l747_74750

/-- A trapezoid inscribed in a circle -/
structure InscribedTrapezoid where
  /-- Radius of the circle -/
  r : ℝ
  /-- Length of each parallel side -/
  a : ℝ
  /-- Length of each non-parallel side -/
  x : ℝ
  /-- The trapezoid is inscribed in the circle -/
  inscribed : True
  /-- The parallel sides are equal -/
  parallel_equal : True
  /-- The non-parallel sides are equal -/
  nonparallel_equal : True

/-- Theorem stating the length of non-parallel sides in the specific trapezoid -/
theorem nonparallel_side_length (t : InscribedTrapezoid) 
  (h1 : t.r = 300) 
  (h2 : t.a = 150) : 
  t.x = 300 := by
  sorry

end nonparallel_side_length_l747_74750


namespace cost_increase_percentage_l747_74742

/-- Proves that given the initial profit is 320% of the cost, and after a cost increase
    (with constant selling price) the profit becomes 66.67% of the selling price,
    then the cost increase percentage is 40%. -/
theorem cost_increase_percentage (C : ℝ) (X : ℝ) : 
  C > 0 →                           -- Assuming positive initial cost
  let S := 4.2 * C                  -- Initial selling price
  let new_profit := 3.2 * C - (X / 100) * C  -- New profit after cost increase
  3.2 * C = 320 / 100 * C →         -- Initial profit is 320% of cost
  new_profit = 2 / 3 * S →          -- New profit is 66.67% of selling price
  X = 40 :=                         -- Cost increase percentage is 40%
by
  sorry


end cost_increase_percentage_l747_74742


namespace complex_number_in_third_quadrant_l747_74787

theorem complex_number_in_third_quadrant :
  let z : ℂ := (1 - Complex.I)^2 / (1 + Complex.I)
  ∃ (a b : ℝ), z = Complex.mk a b ∧ a < 0 ∧ b < 0 :=
by sorry

end complex_number_in_third_quadrant_l747_74787


namespace original_group_size_l747_74763

theorem original_group_size (initial_days : ℕ) (absent_men : ℕ) (final_days : ℕ) : ℕ :=
  let original_size := 42
  let work_amount := original_size * initial_days
  have h1 : work_amount = (original_size - absent_men) * final_days := by sorry
  have h2 : initial_days = 12 := by sorry
  have h3 : absent_men = 6 := by sorry
  have h4 : final_days = 14 := by sorry
  original_size

#check original_group_size

end original_group_size_l747_74763


namespace f_minimum_when_a_is_one_f_nonnegative_iff_a_ge_one_l747_74725

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + 2 / x + a * x - a - 2

theorem f_minimum_when_a_is_one :
  ∃ (min : ℝ), min = 0 ∧ ∀ x > 0, f 1 x ≥ min :=
sorry

theorem f_nonnegative_iff_a_ge_one :
  ∀ a > 0, (∀ x ∈ Set.Icc 1 3, f a x ≥ 0) ↔ a ≥ 1 :=
sorry

end f_minimum_when_a_is_one_f_nonnegative_iff_a_ge_one_l747_74725


namespace divisor_problem_l747_74778

theorem divisor_problem (d : ℕ) (h_pos : d > 0) :
  1200 % d = 3 ∧ 1640 % d = 2 ∧ 1960 % d = 7 → d = 9 ∨ d = 21 := by
  sorry

end divisor_problem_l747_74778


namespace triangle_geometric_sequence_ratio_range_l747_74724

theorem triangle_geometric_sequence_ratio_range (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : b^2 = a*c) : 2 ≤ (b/a + a/b) ∧ (b/a + a/b) < Real.sqrt 5 := by
  sorry

end triangle_geometric_sequence_ratio_range_l747_74724


namespace trapezium_shorter_side_length_l747_74761

theorem trapezium_shorter_side_length 
  (longer_side : ℝ) 
  (height : ℝ) 
  (area : ℝ) 
  (h1 : longer_side = 30) 
  (h2 : height = 16) 
  (h3 : area = 336) : 
  ∃ (shorter_side : ℝ), 
    area = (1 / 2) * (shorter_side + longer_side) * height ∧ 
    shorter_side = 12 := by
  sorry

end trapezium_shorter_side_length_l747_74761


namespace find_y_l747_74737

theorem find_y : ∃ y : ℕ, y^3 * 6^4 / 432 = 5184 ∧ y = 12 := by
  sorry

end find_y_l747_74737


namespace pq_length_is_1098_over_165_l747_74730

/-- The line y = (5/3)x --/
def line1 (x y : ℝ) : Prop := y = (5/3) * x

/-- The line y = (5/12)x --/
def line2 (x y : ℝ) : Prop := y = (5/12) * x

/-- The midpoint of two points --/
def is_midpoint (mx my px py qx qy : ℝ) : Prop :=
  mx = (px + qx) / 2 ∧ my = (py + qy) / 2

/-- The squared distance between two points --/
def distance_squared (x1 y1 x2 y2 : ℝ) : ℝ :=
  (x2 - x1)^2 + (y2 - y1)^2

theorem pq_length_is_1098_over_165 :
  ∀ (px py qx qy : ℝ),
    line1 px py →
    line2 qx qy →
    is_midpoint 10 8 px py qx qy →
    distance_squared px py qx qy = (1098/165)^2 := by
  sorry

end pq_length_is_1098_over_165_l747_74730


namespace smallest_x_divisible_by_3_5_11_l747_74758

theorem smallest_x_divisible_by_3_5_11 : 
  ∃ (x : ℕ), x > 0 ∧ 
  (∀ (y : ℕ), y > 0 → 107 * 151 * y % 3 = 0 ∧ 107 * 151 * y % 5 = 0 ∧ 107 * 151 * y % 11 = 0 → x ≤ y) ∧
  107 * 151 * x % 3 = 0 ∧ 107 * 151 * x % 5 = 0 ∧ 107 * 151 * x % 11 = 0 ∧
  x = 165 := by
  sorry

-- Additional definitions to match the problem conditions
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ (m : ℕ), m > 1 → m < n → n % m ≠ 0

axiom prime_107 : is_prime 107
axiom prime_151 : is_prime 151
axiom prime_3 : is_prime 3
axiom prime_5 : is_prime 5
axiom prime_11 : is_prime 11

end smallest_x_divisible_by_3_5_11_l747_74758


namespace min_value_iff_lower_bound_l747_74711

/-- Given a function f: ℝ → ℝ and a constant M, prove that the following are equivalent:
    1) For all x ∈ ℝ, f(x) ≥ M
    2) M is the minimum value of f -/
theorem min_value_iff_lower_bound (f : ℝ → ℝ) (M : ℝ) :
  (∀ x, f x ≥ M) ↔ (∀ x, f x ≥ M ∧ ∃ y, f y = M) :=
by sorry

end min_value_iff_lower_bound_l747_74711


namespace swimmers_pass_count_l747_74777

/-- Represents a swimmer in the pool --/
structure Swimmer where
  speed : ℝ
  restTime : ℝ

/-- Calculates the number of times swimmers pass each other --/
def countPasses (poolLength : ℝ) (duration : ℝ) (swimmer1 : Swimmer) (swimmer2 : Swimmer) : ℕ :=
  sorry

/-- The main theorem --/
theorem swimmers_pass_count :
  let poolLength : ℝ := 120
  let duration : ℝ := 15 * 60  -- 15 minutes in seconds
  let swimmer1 : Swimmer := { speed := 4, restTime := 30 }
  let swimmer2 : Swimmer := { speed := 3, restTime := 0 }
  countPasses poolLength duration swimmer1 swimmer2 = 17 := by
  sorry

end swimmers_pass_count_l747_74777


namespace gcd_1729_867_l747_74700

theorem gcd_1729_867 : Nat.gcd 1729 867 = 1 := by
  sorry

end gcd_1729_867_l747_74700


namespace susan_ate_six_candies_l747_74796

/-- The number of candies Susan bought on Tuesday -/
def tuesday_candies : ℕ := 3

/-- The number of candies Susan bought on Thursday -/
def thursday_candies : ℕ := 5

/-- The number of candies Susan bought on Friday -/
def friday_candies : ℕ := 2

/-- The number of candies Susan has left -/
def remaining_candies : ℕ := 4

/-- The total number of candies Susan bought -/
def total_candies : ℕ := tuesday_candies + thursday_candies + friday_candies

theorem susan_ate_six_candies : total_candies - remaining_candies = 6 := by
  sorry

end susan_ate_six_candies_l747_74796


namespace range_of_f_l747_74718

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 6*x + 18) / Real.log (1/3)

theorem range_of_f :
  Set.range f = Set.Iic (-2) :=
sorry

end range_of_f_l747_74718


namespace rectangular_prism_ratio_l747_74746

/-- In a rectangular prism with edges a ≤ b ≤ c, if a:b = b:c = c:√(a² + b²), 
    then (a/b)² = (√5 - 1)/2 -/
theorem rectangular_prism_ratio (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a ≤ b) (hbc : b ≤ c) 
  (h_ratio : a / b = b / c ∧ b / c = c / Real.sqrt (a^2 + b^2)) :
  (a / b)^2 = (Real.sqrt 5 - 1) / 2 := by
  sorry

end rectangular_prism_ratio_l747_74746


namespace quadratic_inequality_solution_set_l747_74729

theorem quadratic_inequality_solution_set :
  {x : ℝ | 2 * x^2 - x - 1 > 0} = {x : ℝ | x < -1/2 ∨ x > 1} :=
by sorry

end quadratic_inequality_solution_set_l747_74729


namespace stamps_per_page_l747_74710

theorem stamps_per_page (a b c : ℕ) (ha : a = 1200) (hb : b = 1800) (hc : c = 2400) :
  Nat.gcd a (Nat.gcd b c) = 600 := by
  sorry

end stamps_per_page_l747_74710


namespace set_M_membership_l747_74767

def M : Set ℕ := {x : ℕ | (1 : ℚ) / (x - 2 : ℚ) ≤ 0}

theorem set_M_membership :
  1 ∈ M ∧ 2 ∉ M ∧ 3 ∉ M ∧ 4 ∉ M :=
by sorry

end set_M_membership_l747_74767


namespace min_voters_for_tall_giraffe_l747_74780

/-- Represents the voting structure in the giraffe beauty contest -/
structure VotingStructure where
  total_voters : Nat
  num_districts : Nat
  precincts_per_district : Nat
  voters_per_precinct : Nat

/-- Calculates the minimum number of voters required to win -/
def min_voters_to_win (vs : VotingStructure) : Nat :=
  let districts_to_win := (vs.num_districts + 1) / 2
  let precincts_to_win_per_district := (vs.precincts_per_district + 1) / 2
  let voters_to_win_per_precinct := (vs.voters_per_precinct + 1) / 2
  districts_to_win * precincts_to_win_per_district * voters_to_win_per_precinct

/-- The giraffe beauty contest voting structure -/
def giraffe_contest : VotingStructure :=
  { total_voters := 135
  , num_districts := 5
  , precincts_per_district := 9
  , voters_per_precinct := 3 }

theorem min_voters_for_tall_giraffe :
  min_voters_to_win giraffe_contest = 30 := by
  sorry

#eval min_voters_to_win giraffe_contest

end min_voters_for_tall_giraffe_l747_74780


namespace total_cost_is_55_l747_74745

/-- The total cost of two pairs of shoes, where the first pair costs $22 and the second pair is 50% more expensive than the first pair. -/
def total_cost : ℝ :=
  let first_pair_cost : ℝ := 22
  let second_pair_cost : ℝ := first_pair_cost * 1.5
  first_pair_cost + second_pair_cost

/-- Theorem stating that the total cost of the two pairs of shoes is $55. -/
theorem total_cost_is_55 : total_cost = 55 := by
  sorry

end total_cost_is_55_l747_74745


namespace unique_zero_point_condition_l747_74723

def f (a : ℝ) (x : ℝ) : ℝ := 3 * x^2 + 2 * x - a

theorem unique_zero_point_condition (a : ℝ) :
  (∃! x : ℝ, x ∈ Set.Ioo (-1) 1 ∧ f a x = 0) ↔ (1 < a ∧ a < 5) ∨ a = -1/3 := by
  sorry

end unique_zero_point_condition_l747_74723


namespace platform_length_l747_74789

/-- The length of the platform given a train's characteristics and crossing times. -/
theorem platform_length
  (train_length : ℝ)
  (time_platform : ℝ)
  (time_pole : ℝ)
  (h1 : train_length = 300)
  (h2 : time_platform = 45)
  (h3 : time_pole = 18) :
  let speed := train_length / time_pole
  let total_distance := speed * time_platform
  train_length + (total_distance - train_length) = 450 :=
by sorry

end platform_length_l747_74789


namespace some_magical_beings_are_enchanting_creatures_l747_74779

-- Define the sets
variable (W : Set α) -- Wizards
variable (M : Set α) -- Magical beings
variable (E : Set α) -- Enchanting creatures

-- Define the conditions
variable (h1 : W ⊆ M) -- All wizards are magical beings
variable (h2 : ∃ x, x ∈ E ∩ W) -- Some enchanting creatures are wizards

-- State the theorem
theorem some_magical_beings_are_enchanting_creatures :
  ∃ x, x ∈ M ∩ E := by sorry

end some_magical_beings_are_enchanting_creatures_l747_74779


namespace min_points_per_player_l747_74752

theorem min_points_per_player 
  (num_players : ℕ) 
  (total_points : ℕ) 
  (max_individual_points : ℕ) 
  (h1 : num_players = 12)
  (h2 : total_points = 100)
  (h3 : max_individual_points = 23) :
  ∃ (min_points : ℕ), 
    min_points = 7 ∧ 
    (∃ (scores : List ℕ), 
      scores.length = num_players ∧ 
      scores.sum = total_points ∧
      (∀ s ∈ scores, s ≥ min_points) ∧
      (∃ s ∈ scores, s = max_individual_points) ∧
      (∀ s ∈ scores, s ≤ max_individual_points)) :=
by sorry

end min_points_per_player_l747_74752


namespace cristina_photos_l747_74707

theorem cristina_photos (total_slots : ℕ) (john_photos sarah_photos clarissa_photos : ℕ) 
  (h1 : total_slots = 40)
  (h2 : john_photos = 10)
  (h3 : sarah_photos = 9)
  (h4 : clarissa_photos = 14)
  (h5 : ∃ (cristina_photos : ℕ), cristina_photos + john_photos + sarah_photos + clarissa_photos = total_slots) :
  ∃ (cristina_photos : ℕ), cristina_photos = 7 := by
sorry

end cristina_photos_l747_74707


namespace circle_tangent_problem_l747_74732

-- Define the circles
def Circle := ℝ × ℝ → Prop

-- Define the tangent line
def TangentLine (m : ℝ) (x y : ℝ) : Prop := y = m * x

-- Define the property of being square-free
def SquareFree (n : ℕ) : Prop := ∀ p : ℕ, Prime p → (p^2 ∣ n) → False

-- Define the property of being relatively prime
def RelativelyPrime (a c : ℕ) : Prop := Nat.gcd a c = 1

theorem circle_tangent_problem (C₁ C₂ : Circle) (m : ℝ) (a b c : ℕ) :
  (∃ x y : ℝ, C₁ (x, y) ∧ C₂ (x, y)) →  -- Circles intersect
  C₁ (8, 6) ∧ C₂ (8, 6) →  -- Intersection point at (8,6)
  (∃ r₁ r₂ : ℝ, r₁ * r₂ = 75) →  -- Product of radii is 75
  (∀ x : ℝ, C₁ (x, 0) → x = 0) ∧ (∀ x : ℝ, C₂ (x, 0) → x = 0) →  -- x-axis is tangent
  (∀ x y : ℝ, C₁ (x, y) ∧ TangentLine m x y → x = 0) ∧ 
  (∀ x y : ℝ, C₂ (x, y) ∧ TangentLine m x y → x = 0) →  -- y = mx is tangent
  m > 0 →  -- m is positive
  m = (a : ℝ) * Real.sqrt (b : ℝ) / (c : ℝ) →  -- m in the form a√b/c
  a > 0 ∧ b > 0 ∧ c > 0 →  -- a, b, c are positive
  SquareFree b →  -- b is square-free
  RelativelyPrime a c →  -- a and c are relatively prime
  a + b + c = 282 := by  -- Conclusion
sorry  -- Proof is omitted as per instructions

end circle_tangent_problem_l747_74732


namespace largest_undefined_x_l747_74722

theorem largest_undefined_x : 
  let f (x : ℝ) := 10 * x^2 - 30 * x + 20
  ∃ (max : ℝ), f max = 0 ∧ ∀ x, f x = 0 → x ≤ max :=
by sorry

end largest_undefined_x_l747_74722


namespace final_amount_proof_l747_74776

/-- Calculates the final amount after two years of compound interest with different rates each year. -/
def final_amount (initial : ℝ) (rate1 : ℝ) (rate2 : ℝ) : ℝ :=
  let amount1 := initial * (1 + rate1)
  amount1 * (1 + rate2)

/-- Theorem stating that given the specific initial amount and interest rates, 
    the final amount after two years is as calculated. -/
theorem final_amount_proof :
  final_amount 6552 0.04 0.05 = 7154.784 := by
  sorry

end final_amount_proof_l747_74776


namespace tuna_distribution_l747_74702

theorem tuna_distribution (total_customers : ℕ) (tuna_count : ℕ) (tuna_weight : ℕ) (customers_without_fish : ℕ) :
  total_customers = 100 →
  tuna_count = 10 →
  tuna_weight = 200 →
  customers_without_fish = 20 →
  (tuna_count * tuna_weight) / (total_customers - customers_without_fish) = 25 := by
  sorry

end tuna_distribution_l747_74702


namespace strawberry_pies_count_l747_74781

/-- Given a total number of pies and a ratio for different types of pies,
    calculate the number of pies of a specific type. -/
theorem strawberry_pies_count
  (total_pies : ℕ)
  (apple_ratio blueberry_ratio cherry_ratio strawberry_ratio : ℕ)
  (h_total : total_pies = 48)
  (h_ratios : apple_ratio = 2 ∧ blueberry_ratio = 5 ∧ cherry_ratio = 4 ∧ strawberry_ratio = 1) :
  (strawberry_ratio * total_pies) / (apple_ratio + blueberry_ratio + cherry_ratio + strawberry_ratio) = 4 :=
by sorry

end strawberry_pies_count_l747_74781


namespace round_table_gender_divisibility_l747_74713

theorem round_table_gender_divisibility (n : ℕ) : 
  (∃ k : ℕ, k = n / 2 ∧ k = n - k) → 
  (∃ m : ℕ, n = 4 * m) :=
by sorry

end round_table_gender_divisibility_l747_74713


namespace expression_simplification_l747_74773

theorem expression_simplification :
  ((2 + 3 + 4 + 5) / 2) + ((2 * 5 + 8) / 3) = 13 := by
  sorry

end expression_simplification_l747_74773


namespace conjugate_complex_equation_l747_74771

/-- Two complex numbers are conjugates if their real parts are equal and their imaginary parts are opposites -/
def are_conjugates (a b : ℂ) : Prop := a.re = b.re ∧ a.im = -b.im

/-- The main theorem -/
theorem conjugate_complex_equation (a b : ℂ) :
  are_conjugates a b → (a + b)^2 - 3 * a * b * I = 4 - 6 * I →
  ((a = 1 + I ∧ b = 1 - I) ∨
   (a = -1 - I ∧ b = -1 + I) ∨
   (a = 1 - I ∧ b = 1 + I) ∨
   (a = -1 + I ∧ b = -1 - I)) :=
by sorry

end conjugate_complex_equation_l747_74771


namespace emily_calculation_l747_74772

theorem emily_calculation (n : ℕ) : n = 42 → (n + 1)^2 = n^2 + 85 → (n - 1)^2 = n^2 - 83 := by
  sorry

end emily_calculation_l747_74772


namespace quadratic_equations_solutions_l747_74751

theorem quadratic_equations_solutions :
  (∃ x1 x2 : ℝ, x1^2 + 4*x1 - 2 = 0 ∧ x2^2 + 4*x2 - 2 = 0 ∧ x1 = -2 + Real.sqrt 6 ∧ x2 = -2 - Real.sqrt 6) ∧
  (∃ y1 y2 : ℝ, 2*y1^2 - 3*y1 + 1 = 0 ∧ 2*y2^2 - 3*y2 + 1 = 0 ∧ y1 = 1/2 ∧ y2 = 1) :=
by sorry

end quadratic_equations_solutions_l747_74751


namespace sufficient_but_not_necessary_l747_74703

theorem sufficient_but_not_necessary (x : ℝ) :
  (∀ x, x > 1 → x^2 + 2*x > 0) ∧
  (∃ x, x^2 + 2*x > 0 ∧ x ≤ 1) :=
by sorry

end sufficient_but_not_necessary_l747_74703


namespace factorize_nine_minus_a_squared_l747_74734

theorem factorize_nine_minus_a_squared (a : ℝ) : 9 - a^2 = (3 + a) * (3 - a) := by
  sorry

end factorize_nine_minus_a_squared_l747_74734


namespace region_is_lower_left_l747_74754

-- Define the line
def line (x y : ℝ) : Prop := x - 2*y + 6 = 0

-- Define the region
def region (x y : ℝ) : Prop := x - 2*y + 6 < 0

-- Define what it means to be on the lower left side of the line
def lower_left_side (x y : ℝ) : Prop := x - 2*y + 6 < 0

-- Theorem statement
theorem region_is_lower_left : 
  ∀ (x y : ℝ), region x y → lower_left_side x y :=
sorry

end region_is_lower_left_l747_74754


namespace trip_time_change_l747_74714

/-- Calculates the time required for a trip given the original time, original speed, and new speed -/
def new_trip_time (original_time : ℚ) (original_speed : ℚ) (new_speed : ℚ) : ℚ :=
  (original_time * original_speed) / new_speed

theorem trip_time_change (original_time : ℚ) (original_speed : ℚ) (new_speed : ℚ) 
  (h1 : original_time = 16/3)
  (h2 : original_speed = 80)
  (h3 : new_speed = 50) :
  ∃ (ε : ℚ), abs (new_trip_time original_time original_speed new_speed - 853/100) < ε ∧ ε < 1/100 :=
sorry

end trip_time_change_l747_74714


namespace total_cost_proof_l747_74770

def squat_rack_cost : ℕ := 2500
def barbell_cost_ratio : ℚ := 1 / 10

theorem total_cost_proof :
  squat_rack_cost + (squat_rack_cost : ℚ) * barbell_cost_ratio = 2750 := by
  sorry

end total_cost_proof_l747_74770


namespace point_d_coordinates_l747_74727

/-- Given a line segment AB with endpoints A(-3, 2) and B(5, 10), and a point D on AB
    such that AD = 2DB, and the slope of AB is 1, prove that the coordinates of D are (7/3, 22/3). -/
theorem point_d_coordinates :
  let A : ℝ × ℝ := (-3, 2)
  let B : ℝ × ℝ := (5, 10)
  let D : ℝ × ℝ := (x, y)
  ∀ x y : ℝ,
    (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = (1 - t) • A + t • B) →  -- D is on segment AB
    (x - (-3))^2 + (y - 2)^2 = 4 * ((5 - x)^2 + (10 - y)^2) →  -- AD = 2DB
    (10 - 2) / (5 - (-3)) = 1 →  -- Slope of AB is 1
    D = (7/3, 22/3) :=
by
  sorry

end point_d_coordinates_l747_74727


namespace mashed_potatoes_suggestion_l747_74759

theorem mashed_potatoes_suggestion (bacon : ℕ) (tomatoes : ℕ) (total : ℕ) 
  (h1 : bacon = 374) 
  (h2 : tomatoes = 128) 
  (h3 : total = 826) :
  total - (bacon + tomatoes) = 324 :=
by sorry

end mashed_potatoes_suggestion_l747_74759


namespace ice_cream_volume_l747_74791

/-- The volume of ice cream on a cone -/
theorem ice_cream_volume (h_cone : ℝ) (r : ℝ) (h_cylinder : ℝ)
  (h_cone_pos : h_cone > 0)
  (r_pos : r > 0)
  (h_cylinder_pos : h_cylinder > 0) :
  let v_cone := (1 / 3) * π * r^2 * h_cone
  let v_cylinder := π * r^2 * h_cylinder
  let v_hemisphere := (2 / 3) * π * r^3
  v_cylinder + v_hemisphere = 14.25 * π :=
by
  sorry

#check ice_cream_volume 10 1.5 2

end ice_cream_volume_l747_74791


namespace school_start_time_proof_l747_74708

structure SchoolCommute where
  normalTime : ℕ
  redLightStops : ℕ
  redLightTime : ℕ
  constructionTime : ℕ
  departureTime : Nat × Nat
  lateMinutes : ℕ

def schoolStartTime (c : SchoolCommute) : Nat × Nat :=
  sorry

theorem school_start_time_proof (c : SchoolCommute) 
  (h1 : c.normalTime = 30)
  (h2 : c.redLightStops = 4)
  (h3 : c.redLightTime = 3)
  (h4 : c.constructionTime = 10)
  (h5 : c.departureTime = (7, 15))
  (h6 : c.lateMinutes = 7) :
  schoolStartTime c = (8, 0) :=
by
  sorry

end school_start_time_proof_l747_74708


namespace tan_2_implies_sum_23_10_l747_74735

theorem tan_2_implies_sum_23_10 (θ : Real) (h : Real.tan θ = 2) :
  (Real.sin θ + Real.cos θ) / Real.sin θ + Real.sin θ ^ 2 = 23 / 10 := by
  sorry

end tan_2_implies_sum_23_10_l747_74735


namespace expand_expression_l747_74728

theorem expand_expression (x y : ℝ) : (3*x - 5) * (4*y + 20) = 12*x*y + 60*x - 20*y - 100 := by
  sorry

end expand_expression_l747_74728


namespace dice_product_probability_composite_probability_l747_74741

/-- A function that determines if a number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- The set of possible outcomes when rolling a 6-sided die -/
def dieOutcomes : Finset ℕ := sorry

/-- The set of all possible outcomes when rolling 4 dice -/
def allOutcomes : Finset (ℕ × ℕ × ℕ × ℕ) := sorry

/-- The product of the numbers in a 4-tuple -/
def product (t : ℕ × ℕ × ℕ × ℕ) : ℕ := sorry

/-- The set of outcomes that result in a non-composite product -/
def nonCompositeOutcomes : Finset (ℕ × ℕ × ℕ × ℕ) := sorry

theorem dice_product_probability :
  (Finset.card nonCompositeOutcomes : ℚ) / (Finset.card allOutcomes : ℚ) = 13 / 1296 :=
sorry

theorem composite_probability :
  1 - (Finset.card nonCompositeOutcomes : ℚ) / (Finset.card allOutcomes : ℚ) = 1283 / 1296 :=
sorry

end dice_product_probability_composite_probability_l747_74741


namespace even_sum_probability_l747_74757

theorem even_sum_probability (wheel1_even_prob wheel2_even_prob : ℚ) 
  (h1 : wheel1_even_prob = 3 / 5)
  (h2 : wheel2_even_prob = 1 / 2) : 
  wheel1_even_prob * wheel2_even_prob + (1 - wheel1_even_prob) * (1 - wheel2_even_prob) = 1 / 2 := by
  sorry

end even_sum_probability_l747_74757


namespace apartment_building_occupancy_l747_74739

theorem apartment_building_occupancy :
  let total_floors : ℕ := 12
  let full_floors : ℕ := total_floors / 2
  let half_capacity_floors : ℕ := total_floors - full_floors
  let apartments_per_floor : ℕ := 10
  let people_per_apartment : ℕ := 4
  let people_per_full_floor : ℕ := apartments_per_floor * people_per_apartment
  let people_per_half_floor : ℕ := people_per_full_floor / 2
  let total_people : ℕ := full_floors * people_per_full_floor + half_capacity_floors * people_per_half_floor
  total_people = 360 := by
  sorry

end apartment_building_occupancy_l747_74739


namespace trapezium_circle_radius_l747_74719

/-- Represents a trapezium PQRS with a circle tangent to all sides -/
structure TrapeziumWithCircle where
  -- Length of PQ and SR
  side_length : ℝ
  -- Area of the trapezium
  area : ℝ
  -- Assertion that SP is parallel to RQ
  sp_parallel_rq : Prop
  -- Assertion that all sides are tangent to the circle
  all_sides_tangent : Prop

/-- The radius of the circle in a trapezium with given properties -/
def circle_radius (t : TrapeziumWithCircle) : ℝ :=
  12

/-- Theorem stating that for a trapezium with given properties, the radius of the inscribed circle is 12 -/
theorem trapezium_circle_radius 
  (t : TrapeziumWithCircle) 
  (h1 : t.side_length = 25)
  (h2 : t.area = 600) :
  circle_radius t = 12 := by sorry

end trapezium_circle_radius_l747_74719


namespace candy_problem_l747_74793

def candy_remaining (initial : ℕ) (day : ℕ) : ℚ :=
  match day with
  | 0 => initial
  | 1 => initial / 2
  | 2 => initial / 2 * (1 / 3)
  | 3 => initial / 2 * (1 / 3) * (1 / 4)
  | 4 => initial / 2 * (1 / 3) * (1 / 4) * (1 / 5)
  | 5 => initial / 2 * (1 / 3) * (1 / 4) * (1 / 5) * (1 / 6)
  | _ => 0

theorem candy_problem (initial : ℕ) :
  candy_remaining initial 5 = 1 ↔ initial = 720 := by
  sorry

end candy_problem_l747_74793


namespace probability_of_white_ball_l747_74785

theorem probability_of_white_ball 
  (prob_red : ℝ) 
  (prob_black : ℝ) 
  (h1 : prob_red = 0.4) 
  (h2 : prob_black = 0.25) 
  (h3 : prob_red + prob_black + (1 - prob_red - prob_black) = 1) :
  1 - prob_red - prob_black = 0.35 :=
by sorry

end probability_of_white_ball_l747_74785


namespace point_on_circle_x_value_l747_74706

/-- A circle in the xy-plane with diameter endpoints (-3,0) and (21,0) -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  h1 : center = (9, 0)
  h2 : radius = 12

/-- A point on the circle -/
structure PointOnCircle (c : Circle) where
  x : ℝ
  y : ℝ
  h : (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

theorem point_on_circle_x_value (c : Circle) (p : PointOnCircle c) (h : p.y = 12) :
  p.x = 9 := by
  sorry

end point_on_circle_x_value_l747_74706


namespace shaded_fraction_of_rectangle_l747_74704

theorem shaded_fraction_of_rectangle : 
  let rectangle_length : ℝ := 10
  let rectangle_width : ℝ := 20
  let total_area : ℝ := rectangle_length * rectangle_width
  let quarter_area : ℝ := total_area / 4
  let shaded_area : ℝ := quarter_area / 2
  shaded_area / total_area = 1 / 8 := by sorry

end shaded_fraction_of_rectangle_l747_74704
