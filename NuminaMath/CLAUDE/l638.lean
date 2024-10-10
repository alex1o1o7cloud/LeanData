import Mathlib

namespace vinegar_left_is_60_l638_63867

/-- Represents the pickle-making scenario with given supplies and rules. -/
structure PickleScenario where
  jars : ℕ
  cucumbers : ℕ
  initial_vinegar : ℕ
  pickles_per_cucumber : ℕ
  pickles_per_jar : ℕ
  vinegar_per_jar : ℕ

/-- Calculates the amount of vinegar left after making pickles. -/
def vinegar_left (scenario : PickleScenario) : ℕ :=
  let total_pickles := scenario.cucumbers * scenario.pickles_per_cucumber
  let max_jarred_pickles := scenario.jars * scenario.pickles_per_jar
  let actual_jarred_pickles := min total_pickles max_jarred_pickles
  let jars_used := actual_jarred_pickles / scenario.pickles_per_jar
  let vinegar_used := jars_used * scenario.vinegar_per_jar
  scenario.initial_vinegar - vinegar_used

/-- Theorem stating that given the specific scenario, 60 oz of vinegar will be left. -/
theorem vinegar_left_is_60 :
  let scenario : PickleScenario := {
    jars := 4,
    cucumbers := 10,
    initial_vinegar := 100,
    pickles_per_cucumber := 6,
    pickles_per_jar := 12,
    vinegar_per_jar := 10
  }
  vinegar_left scenario = 60 := by sorry

end vinegar_left_is_60_l638_63867


namespace compound_interest_rate_l638_63883

theorem compound_interest_rate (P : ℝ) (t : ℕ) (CI : ℝ) (r : ℝ) : 
  P = 4500 →
  t = 2 →
  CI = 945.0000000000009 →
  (P + CI) = P * (1 + r) ^ t →
  r = 0.1 := by
  sorry

end compound_interest_rate_l638_63883


namespace group_communication_l638_63849

theorem group_communication (n k : ℕ) : 
  n > 0 → 
  k > 0 → 
  k * (n - 1) * n = 440 → 
  n = 11 := by
sorry

end group_communication_l638_63849


namespace money_share_difference_l638_63898

theorem money_share_difference (total : ℝ) (moses_percent : ℝ) (rachel_percent : ℝ) 
  (h1 : total = 80)
  (h2 : moses_percent = 0.35)
  (h3 : rachel_percent = 0.20) : 
  moses_percent * total - (total - (moses_percent * total + rachel_percent * total)) / 2 = 10 := by
  sorry

end money_share_difference_l638_63898


namespace arctan_equation_solution_l638_63870

theorem arctan_equation_solution (x : ℝ) :
  Real.arctan (1 / x) + Real.arctan (1 / x^2) = π / 3 →
  x = (1 + Real.sqrt (13 + 4 * Real.sqrt 3)) / (2 * Real.sqrt 3) ∨
  x = (1 - Real.sqrt (13 + 4 * Real.sqrt 3)) / (2 * Real.sqrt 3) :=
by sorry

end arctan_equation_solution_l638_63870


namespace tims_dimes_count_l638_63842

/-- Represents the number of coins of each type --/
structure CoinCount where
  nickels : ℕ
  dimes : ℕ
  halfDollars : ℕ

/-- Calculates the total value of coins in dollars --/
def coinValue (c : CoinCount) : ℚ :=
  0.05 * c.nickels + 0.10 * c.dimes + 0.50 * c.halfDollars

/-- Represents Tim's earnings from shining shoes and tips --/
structure TimsEarnings where
  shoeShining : CoinCount
  tipJar : CoinCount

/-- The main theorem to prove --/
theorem tims_dimes_count 
  (earnings : TimsEarnings)
  (h1 : earnings.shoeShining.nickels = 3)
  (h2 : earnings.tipJar.dimes = 7)
  (h3 : earnings.tipJar.halfDollars = 9)
  (h4 : coinValue earnings.shoeShining + coinValue earnings.tipJar = 6.65) :
  earnings.shoeShining.dimes = 13 :=
by sorry

end tims_dimes_count_l638_63842


namespace necessary_not_sufficient_l638_63839

theorem necessary_not_sufficient (a b c : ℝ) :
  (∀ a b c : ℝ, a * c^2 > b * c^2 → a > b ∨ c = 0) ∧
  (∃ a b c : ℝ, a * c^2 > b * c^2 ∧ a ≤ b) :=
sorry

end necessary_not_sufficient_l638_63839


namespace songs_downloaded_l638_63868

def internet_speed : ℝ := 20
def download_time : ℝ := 0.5
def song_size : ℝ := 5

theorem songs_downloaded : 
  ⌊(internet_speed * download_time * 3600) / song_size⌋ = 7200 := by sorry

end songs_downloaded_l638_63868


namespace quadratic_function_k_l638_63881

/-- Quadratic function g(x) = ax^2 + bx + c -/
def g (a b c : ℤ) (x : ℚ) : ℚ := a * x^2 + b * x + c

theorem quadratic_function_k (a b c k : ℤ) : 
  g a b c (-1) = 0 → 
  (30 < g a b c 5 ∧ g a b c 5 < 40) → 
  (120 < g a b c 7 ∧ g a b c 7 < 130) → 
  (2000 * k < g a b c 50 ∧ g a b c 50 < 2000 * (k + 1)) → 
  k = 5 := by
  sorry

end quadratic_function_k_l638_63881


namespace nectar_water_percentage_l638_63852

/-- The ratio of flower-nectar to honey produced -/
def nectarToHoneyRatio : ℝ := 1.6

/-- The percentage of water in the produced honey -/
def honeyWaterPercentage : ℝ := 20

/-- The percentage of water in flower-nectar -/
def nectarWaterPercentage : ℝ := 50

theorem nectar_water_percentage :
  nectarWaterPercentage = 100 * (nectarToHoneyRatio - (1 - honeyWaterPercentage / 100)) / nectarToHoneyRatio :=
by sorry

end nectar_water_percentage_l638_63852


namespace octal_to_decimal_fraction_l638_63811

theorem octal_to_decimal_fraction (c d : ℕ) : 
  (543 : ℕ) = 5 * 8^2 + 4 * 8^1 + 3 * 8^0 →
  (2 * 10 + c) * 10 + d = 5 * 8^2 + 4 * 8^1 + 3 * 8^0 →
  0 ≤ c ∧ c ≤ 9 →
  0 ≤ d ∧ d ≤ 9 →
  (c * d : ℚ) / 12 = 5 / 4 :=
by sorry

end octal_to_decimal_fraction_l638_63811


namespace proportion_solution_l638_63841

/-- Given a proportion x : 10 :: 8 : 0.6, prove that x = 400/3 -/
theorem proportion_solution (x : ℚ) : (x / 10 = 8 / (3/5)) → x = 400/3 := by
  sorry

end proportion_solution_l638_63841


namespace ethanol_in_tank_l638_63860

/-- Calculates the total amount of ethanol in a fuel tank -/
def total_ethanol (tank_capacity : ℝ) (fuel_a_volume : ℝ) (fuel_a_ethanol_percent : ℝ) (fuel_b_ethanol_percent : ℝ) : ℝ :=
  let fuel_b_volume := tank_capacity - fuel_a_volume
  let ethanol_a := fuel_a_volume * fuel_a_ethanol_percent
  let ethanol_b := fuel_b_volume * fuel_b_ethanol_percent
  ethanol_a + ethanol_b

/-- Theorem stating that the total ethanol in the given scenario is 30 gallons -/
theorem ethanol_in_tank : 
  total_ethanol 204 66 0.12 0.16 = 30 := by
  sorry

#eval total_ethanol 204 66 0.12 0.16

end ethanol_in_tank_l638_63860


namespace reciprocal_equality_implies_equality_l638_63809

theorem reciprocal_equality_implies_equality (x y : ℝ) (h : x ≠ 0) (k : y ≠ 0) : 
  1 / x = 1 / y → x = y := by
sorry

end reciprocal_equality_implies_equality_l638_63809


namespace reader_count_l638_63859

/-- The number of readers who read science fiction -/
def science_fiction_readers : ℕ := 120

/-- The number of readers who read literary works -/
def literary_works_readers : ℕ := 90

/-- The number of readers who read both science fiction and literary works -/
def both_genres_readers : ℕ := 60

/-- The total number of readers in the group -/
def total_readers : ℕ := science_fiction_readers + literary_works_readers - both_genres_readers

theorem reader_count : total_readers = 150 := by
  sorry

end reader_count_l638_63859


namespace sallys_cards_l638_63812

/-- The number of Pokemon cards Sally had initially -/
def initial_cards : ℕ := 27

/-- The number of cards Dan gave to Sally -/
def dans_cards : ℕ := 41

/-- The number of cards Sally bought -/
def bought_cards : ℕ := 20

/-- The total number of cards Sally has now -/
def total_cards : ℕ := 88

/-- Theorem stating that the initial number of cards plus the acquired cards equals the total cards -/
theorem sallys_cards : initial_cards + dans_cards + bought_cards = total_cards := by
  sorry

end sallys_cards_l638_63812


namespace sum_of_solutions_is_four_l638_63832

theorem sum_of_solutions_is_four : ∃ (S : Finset Int), 
  (∀ x : Int, x ∈ S ↔ x^2 = 192 + x) ∧ (S.sum id = 4) := by
  sorry

end sum_of_solutions_is_four_l638_63832


namespace unique_orthocenter_line_l638_63851

/-- The ellipse with equation x^2/2 + y^2 = 1 -/
def ellipse (x y : ℝ) : Prop := x^2/2 + y^2 = 1

/-- The upper vertex of the ellipse -/
def B : ℝ × ℝ := (0, 1)

/-- The right focus of the ellipse -/
def F : ℝ × ℝ := (1, 0)

/-- A line that intersects the ellipse -/
def line_intersects_ellipse (m b : ℝ) : Prop :=
  ∃ x₁ x₂ y₁ y₂ : ℝ, x₁ ≠ x₂ ∧ 
    ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧
    y₁ = m * x₁ + b ∧ y₂ = m * x₂ + b

/-- F is the orthocenter of triangle BMN -/
def F_is_orthocenter (M N : ℝ × ℝ) : Prop :=
  let (xm, ym) := M
  let (xn, yn) := N
  (1 - xn) * xm - yn * (ym - 1) = 0 ∧
  (1 - xm) * xn - ym * (yn - 1) = 0

theorem unique_orthocenter_line :
  ∃! m b : ℝ, 
    line_intersects_ellipse m b ∧
    (∀ M N : ℝ × ℝ, 
      ellipse M.1 M.2 → ellipse N.1 N.2 → 
      M.2 = m * M.1 + b → N.2 = m * N.1 + b →
      F_is_orthocenter M N) ∧
    m = 1 ∧ b = -4/3 :=
sorry

end unique_orthocenter_line_l638_63851


namespace doll_factory_operation_time_l638_63863

/-- Calculate the total machine operation time for dolls and accessories -/
theorem doll_factory_operation_time :
  let num_dolls : ℕ := 12000
  let shoes_per_doll : ℕ := 2
  let bags_per_doll : ℕ := 3
  let cosmetics_per_doll : ℕ := 1
  let hats_per_doll : ℕ := 5
  let doll_production_time : ℕ := 45
  let accessory_production_time : ℕ := 10

  let total_accessories : ℕ := num_dolls * (shoes_per_doll + bags_per_doll + cosmetics_per_doll + hats_per_doll)
  let doll_time : ℕ := num_dolls * doll_production_time
  let accessory_time : ℕ := total_accessories * accessory_production_time
  let total_time : ℕ := doll_time + accessory_time

  total_time = 1860000 := by
  sorry

end doll_factory_operation_time_l638_63863


namespace matching_socks_probability_l638_63856

def black_socks : ℕ := 12
def white_socks : ℕ := 6
def blue_socks : ℕ := 9

def total_socks : ℕ := black_socks + white_socks + blue_socks

def matching_pairs : ℕ := (black_socks.choose 2) + (white_socks.choose 2) + (blue_socks.choose 2)

def total_pairs : ℕ := total_socks.choose 2

theorem matching_socks_probability :
  (matching_pairs : ℚ) / total_pairs = 1 / 3 := by sorry

end matching_socks_probability_l638_63856


namespace area_of_quadrilateral_l638_63874

/-- Given a rectangle ACDE with AC = 48 and AE = 30, where point B divides AC in ratio 1:3
    and point F divides AE in ratio 2:3, the area of quadrilateral ABDF is equal to 468. -/
theorem area_of_quadrilateral (AC AE : ℝ) (B F : ℝ) : 
  AC = 48 → 
  AE = 30 → 
  B / AC = 1 / 4 → 
  F / AE = 2 / 5 → 
  (AC * AE) - (3 * AC * AE / 4) - (3 * AC * AE / 5) = 468 := by
  sorry

end area_of_quadrilateral_l638_63874


namespace negation_of_universal_proposition_l638_63880

def A : Set ℤ := {x | ∃ k, x = 2*k + 1}
def B : Set ℤ := {x | ∃ k, x = 2*k}

theorem negation_of_universal_proposition :
  (¬ (∀ x ∈ A, (2 * x) ∈ B)) ↔ (∃ x ∈ A, (2 * x) ∉ B) :=
sorry

end negation_of_universal_proposition_l638_63880


namespace isosceles_triangle_base_length_l638_63834

/-- An isosceles triangle with two sides of length 8 cm and perimeter 30 cm has a base of length 14 cm. -/
theorem isosceles_triangle_base_length :
  ∀ (base_length : ℝ),
    base_length > 0 →
    2 * 8 + base_length = 30 →
    base_length = 14 :=
by sorry

end isosceles_triangle_base_length_l638_63834


namespace quadratic_properties_l638_63818

/-- Represents a quadratic function ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate that checks if x is in the solution set (2, 3) -/
def inSolutionSet (x : ℝ) : Prop := 2 < x ∧ x < 3

/-- The quadratic function is positive in the interval (2, 3) -/
def isPositiveInInterval (f : QuadraticFunction) : Prop :=
  ∀ x, inSolutionSet x → f.a * x^2 + f.b * x + f.c > 0

theorem quadratic_properties (f : QuadraticFunction) 
  (h : isPositiveInInterval f) : 
  f.a < 0 ∧ f.b * f.c < 0 ∧ f.b + f.c = f.a ∧ f.a - f.b + f.c ≤ 0 := by
  sorry

end quadratic_properties_l638_63818


namespace painted_cube_equality_l638_63802

/-- Represents a cube with edge length n, painted with alternating colors on adjacent faces. -/
structure PaintedCube where
  n : ℕ
  n_gt_two : n > 2

/-- The number of unit cubes with exactly one black face in a painted cube. -/
def black_face_count (cube : PaintedCube) : ℕ :=
  3 * (cube.n - 2)^2

/-- The number of unpainted unit cubes in a painted cube. -/
def unpainted_count (cube : PaintedCube) : ℕ :=
  (cube.n - 2)^3

/-- Theorem stating that the number of unit cubes with exactly one black face
    equals the number of unpainted unit cubes if and only if n = 5. -/
theorem painted_cube_equality (cube : PaintedCube) :
  black_face_count cube = unpainted_count cube ↔ cube.n = 5 := by
  sorry

end painted_cube_equality_l638_63802


namespace ellipse_max_major_axis_l638_63835

theorem ellipse_max_major_axis 
  (a b : ℝ) 
  (h_ab : a > b ∧ b > 0) 
  (e : ℝ) 
  (h_e : e ∈ Set.Icc (1/2) (Real.sqrt 2 / 2)) 
  (h_perp : ∀ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁^2/a^2 + y₁^2/b^2 = 1 → 
    y₁ = -x₁ + 1 → 
    x₂^2/a^2 + y₂^2/b^2 = 1 → 
    y₂ = -x₂ + 1 → 
    x₁*x₂ + y₁*y₂ = 0) 
  (h_ecc : e^2 = 1 - b^2/a^2) :
  ∃ (max_axis : ℝ), max_axis = Real.sqrt 6 ∧ 
    ∀ (axis : ℝ), axis = 2*a → axis ≤ max_axis :=
sorry

end ellipse_max_major_axis_l638_63835


namespace sum_of_roots_quadratic_l638_63806

theorem sum_of_roots_quadratic (x : ℝ) : 
  let a : ℝ := 3
  let b : ℝ := -12
  let c : ℝ := 12
  let sum_of_roots := -b / a
  (3 * x^2 - 12 * x + 12 = 0) → sum_of_roots = 4 := by
sorry

end sum_of_roots_quadratic_l638_63806


namespace not_all_six_multiples_have_prime_neighbor_l638_63847

theorem not_all_six_multiples_have_prime_neighbor :
  ∃ n : ℕ, 6 ∣ n ∧ ¬(Nat.Prime (n - 1) ∨ Nat.Prime (n + 1)) := by
  sorry

end not_all_six_multiples_have_prime_neighbor_l638_63847


namespace new_year_gift_exchange_l638_63888

/-- Represents a group of friends exchanging gifts -/
structure GiftExchange where
  num_friends : Nat
  num_exchanges : Nat

/-- Predicate to check if the number of friends receiving 4 gifts is valid -/
def valid_four_gift_recipients (ge : GiftExchange) (n : Nat) : Prop :=
  n = 2 ∨ n = 4

/-- Theorem stating that in the given scenario, the number of friends receiving 4 gifts is either 2 or 4 -/
theorem new_year_gift_exchange (ge : GiftExchange) 
  (h1 : ge.num_friends = 6)
  (h2 : ge.num_exchanges = 13) :
  ∃ n : Nat, valid_four_gift_recipients ge n := by
  sorry

end new_year_gift_exchange_l638_63888


namespace geometric_sequence_m_value_l638_63845

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_m_value
  (a : ℕ → ℝ)
  (q : ℝ)
  (h1 : geometric_sequence a q)
  (h2 : q ≠ 1 ∧ q ≠ -1)
  (h3 : a 1 = -1)
  (h4 : ∃ m : ℕ, a m = a 1 * a 2 * a 3 * a 4 * a 5) :
  ∃ m : ℕ, m = 11 ∧ a m = a 1 * a 2 * a 3 * a 4 * a 5 :=
sorry

end geometric_sequence_m_value_l638_63845


namespace second_year_increase_is_fifteen_percent_l638_63896

/-- Calculates the percentage increase in the second year given the initial population,
    first year increase percentage, and final population after two years. -/
def second_year_increase (initial_population : ℕ) (first_year_increase : ℚ) (final_population : ℕ) : ℚ :=
  let population_after_first_year := initial_population * (1 + first_year_increase)
  ((final_population : ℚ) / population_after_first_year - 1) * 100

/-- Theorem stating that given the specific conditions of the problem,
    the second year increase is 15%. -/
theorem second_year_increase_is_fifteen_percent :
  second_year_increase 800 (25 / 100) 1150 = 15 := by
  sorry

#eval second_year_increase 800 (25 / 100) 1150

end second_year_increase_is_fifteen_percent_l638_63896


namespace day_250_is_tuesday_l638_63891

/-- Days of the week represented as integers mod 7 -/
inductive DayOfWeek : Type
| Sunday : DayOfWeek
| Monday : DayOfWeek
| Tuesday : DayOfWeek
| Wednesday : DayOfWeek
| Thursday : DayOfWeek
| Friday : DayOfWeek
| Saturday : DayOfWeek

def dayOfWeek (n : Nat) : DayOfWeek :=
  match n % 7 with
  | 0 => DayOfWeek.Sunday
  | 1 => DayOfWeek.Monday
  | 2 => DayOfWeek.Tuesday
  | 3 => DayOfWeek.Wednesday
  | 4 => DayOfWeek.Thursday
  | 5 => DayOfWeek.Friday
  | _ => DayOfWeek.Saturday

theorem day_250_is_tuesday (h : dayOfWeek 35 = DayOfWeek.Wednesday) :
  dayOfWeek 250 = DayOfWeek.Tuesday := by
  sorry

end day_250_is_tuesday_l638_63891


namespace intersecting_squares_area_difference_l638_63850

/-- Given four intersecting squares with sides 12, 9, 7, and 3,
    the difference between the sum of the areas of the largest and third largest squares
    and the sum of the areas of the second largest and smallest squares is 103. -/
theorem intersecting_squares_area_difference : 
  let a := 12 -- side length of the largest square
  let b := 9  -- side length of the second largest square
  let c := 7  -- side length of the third largest square
  let d := 3  -- side length of the smallest square
  (a ^ 2 + c ^ 2) - (b ^ 2 + d ^ 2) = 103 := by
sorry

#eval (12 ^ 2 + 7 ^ 2) - (9 ^ 2 + 3 ^ 2) -- This should evaluate to 103

end intersecting_squares_area_difference_l638_63850


namespace equation_positive_root_l638_63861

theorem equation_positive_root (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ (6 / (x - 2) - 1 = a * x / (2 - x))) → a = -3 := by
  sorry

end equation_positive_root_l638_63861


namespace fraction_product_equals_one_l638_63820

theorem fraction_product_equals_one : 
  (36 : ℚ) / 34 * 26 / 48 * 136 / 78 = 1 := by sorry

end fraction_product_equals_one_l638_63820


namespace circle_points_l638_63821

theorem circle_points (π : ℝ) (h : π > 0) : 
  let radii : List ℝ := [1.5, 2, 3.5, 4.5, 5.5]
  let circumference (r : ℝ) := 2 * π * r
  let area (r : ℝ) := π * r^2
  let points := radii.map (λ r => (circumference r, area r))
  points = [(3*π, 2.25*π), (4*π, 4*π), (7*π, 12.25*π), (9*π, 20.25*π), (11*π, 30.25*π)] := by
  sorry

end circle_points_l638_63821


namespace coin_flip_probability_l638_63815

theorem coin_flip_probability (p : ℝ) (h : p = 1 / 2) :
  p * p * (1 - p) * (1 - p) = 1 / 16 := by
  sorry

end coin_flip_probability_l638_63815


namespace system_solution_l638_63830

theorem system_solution (x y a : ℝ) : 
  x - 2*y = a - 6 →
  2*x + 5*y = 2*a →
  x + y = 9 →
  a = 11 := by sorry

end system_solution_l638_63830


namespace bill_donut_order_ways_l638_63893

/-- The number of ways to distribute identical items into distinct groups -/
def distribute_items (items : ℕ) (groups : ℕ) : ℕ :=
  Nat.choose (items + groups - 1) (groups - 1)

/-- The number of ways Bill can fulfill his donut order -/
theorem bill_donut_order_ways : distribute_items 3 4 = 20 := by
  sorry

end bill_donut_order_ways_l638_63893


namespace sequence_length_l638_63846

/-- The number of terms in the sequence 1, 2³, 2⁶, 2⁹, ..., 2³ⁿ⁺⁶ -/
def num_terms (n : ℕ) : ℕ := n + 3

/-- The exponent of the k-th term in the sequence -/
def exponent (k : ℕ) : ℕ := 3 * (k - 1)

theorem sequence_length (n : ℕ) : 
  (∃ k : ℕ, k > 0 ∧ exponent k = 3 * n + 6) → 
  num_terms n = (Finset.range (n + 3)).card :=
sorry

end sequence_length_l638_63846


namespace sphere_radius_in_truncated_cone_l638_63824

/-- A truncated cone with a tangent sphere -/
structure TruncatedConeWithSphere where
  bottom_radius : ℝ
  top_radius : ℝ
  sphere_radius : ℝ
  tangent_to_top : Bool
  tangent_to_bottom : Bool
  tangent_to_lateral : Bool

/-- The theorem stating the radius of the sphere in a specific truncated cone configuration -/
theorem sphere_radius_in_truncated_cone
  (cone : TruncatedConeWithSphere)
  (h1 : cone.bottom_radius = 12)
  (h2 : cone.top_radius = 3)
  (h3 : cone.tangent_to_top = true)
  (h4 : cone.tangent_to_bottom = true)
  (h5 : cone.tangent_to_lateral = true) :
  cone.sphere_radius = 6 := by
  sorry

end sphere_radius_in_truncated_cone_l638_63824


namespace linear_function_decreases_iff_positive_slope_l638_63804

/-- A linear function represented by its slope and y-intercept -/
structure LinearFunction where
  slope : ℝ
  intercept : ℝ

/-- The value of a linear function at a given x -/
def LinearFunction.value (f : LinearFunction) (x : ℝ) : ℝ :=
  f.slope * x + f.intercept

/-- A linear function decreases as x decreases iff its slope is positive -/
theorem linear_function_decreases_iff_positive_slope (f : LinearFunction) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f.value x₁ < f.value x₂) ↔ f.slope > 0 := by
  sorry

end linear_function_decreases_iff_positive_slope_l638_63804


namespace gcd_digits_bound_l638_63838

theorem gcd_digits_bound (a b : ℕ) : 
  100000 ≤ a ∧ a < 1000000 ∧ 
  100000 ≤ b ∧ b < 1000000 ∧ 
  1000000000 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10000000000 →
  Nat.gcd a b < 1000 :=
by sorry

end gcd_digits_bound_l638_63838


namespace no_special_eight_digit_number_l638_63890

theorem no_special_eight_digit_number : ¬∃ N : ℕ,
  (10000000 ≤ N ∧ N < 100000000) ∧
  (∀ i : Fin 8, 
    let digit := (N / (10 ^ (7 - i.val))) % 10
    digit ≠ 0 ∧
    N % digit = i.val + 1) :=
sorry

end no_special_eight_digit_number_l638_63890


namespace no_solution_to_inequality_system_l638_63865

theorem no_solution_to_inequality_system :
  ¬∃ x : ℝ, (x / 6 + 7 / 2 > (3 * x + 29) / 5) ∧
            (x + 9 / 2 > x / 8) ∧
            (11 / 3 - x / 6 < (34 - 3 * x) / 5) := by
  sorry

end no_solution_to_inequality_system_l638_63865


namespace right_triangle_side_length_l638_63899

theorem right_triangle_side_length 
  (Q R S : ℝ × ℝ) 
  (right_angle_Q : (R.1 - Q.1) * (S.1 - Q.1) + (R.2 - Q.2) * (S.2 - Q.2) = 0) 
  (cos_R : (R.1 - Q.1) / Real.sqrt ((R.1 - S.1)^2 + (R.2 - S.2)^2) = 5/13) 
  (RS_length : (R.1 - S.1)^2 + (R.2 - S.2)^2 = 13^2) : 
  (Q.1 - S.1)^2 + (Q.2 - S.2)^2 = 12^2 := by
sorry


end right_triangle_side_length_l638_63899


namespace product_sum_relation_l638_63878

theorem product_sum_relation (a b : ℝ) : 
  (a * b = 2 * (a + b) + 12) → (b = 10) → (b - a = 6) := by
  sorry

end product_sum_relation_l638_63878


namespace smallest_number_divisible_after_increase_l638_63876

theorem smallest_number_divisible_after_increase : ∃ (k : ℕ), 
  (∀ (n : ℕ), n < 3153 → ¬∃ (m : ℕ), (n + m) % 18 = 0 ∧ (n + m) % 70 = 0 ∧ (n + m) % 25 = 0 ∧ (n + m) % 21 = 0) ∧
  (∃ (m : ℕ), (3153 + m) % 18 = 0 ∧ (3153 + m) % 70 = 0 ∧ (3153 + m) % 25 = 0 ∧ (3153 + m) % 21 = 0) :=
by
  sorry

end smallest_number_divisible_after_increase_l638_63876


namespace geometric_series_ratio_l638_63879

theorem geometric_series_ratio (a : ℝ) (r : ℝ) : 
  (∃ (S : ℝ), S = a / (1 - r) ∧ S = 24) →
  (∃ (S_odd : ℝ), S_odd = a * r / (1 - r^2) ∧ S_odd = 8) →
  r = 1/2 := by
sorry

end geometric_series_ratio_l638_63879


namespace parsley_rows_juvy_parsley_rows_l638_63831

/-- Calculates the number of rows planted with parsley in Juvy's garden. -/
theorem parsley_rows (total_rows : Nat) (plants_per_row : Nat) (rosemary_rows : Nat) (chives_count : Nat) : Nat :=
  let remaining_rows := total_rows - rosemary_rows
  let chives_rows := chives_count / plants_per_row
  remaining_rows - chives_rows

/-- Proves that Juvy plants parsley in 3 rows given the garden's conditions. -/
theorem juvy_parsley_rows : parsley_rows 20 10 2 150 = 3 := by
  sorry

end parsley_rows_juvy_parsley_rows_l638_63831


namespace g_three_properties_l638_63885

/-- A function satisfying the given condition for all real x and y -/
def special_function (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, g x * g y - g (x * y) = x + y + 1

/-- The theorem stating the properties of g(3) -/
theorem g_three_properties (g : ℝ → ℝ) (h : special_function g) :
  (∃ a b : ℝ, (∀ x : ℝ, g 3 = x → (x = a ∨ x = b)) ∧ a + b = 0) :=
sorry

end g_three_properties_l638_63885


namespace locus_proof_methods_correctness_l638_63866

-- Define a type for points in a geometric space
variable {Point : Type}

-- Define a predicate for points satisfying the locus conditions
variable (satisfiesConditions : Point → Prop)

-- Define a predicate for points being on the locus
variable (onLocus : Point → Prop)

-- Define the correctness of each statement
def statementA : Prop :=
  (∀ p : Point, onLocus p → satisfiesConditions p) ∧
  (∀ p : Point, ¬onLocus p → ¬satisfiesConditions p)

def statementB : Prop :=
  (∀ p : Point, ¬satisfiesConditions p → onLocus p) ∧
  (∀ p : Point, onLocus p → satisfiesConditions p)

def statementC : Prop :=
  (∀ p : Point, satisfiesConditions p → onLocus p) ∧
  (∀ p : Point, ¬onLocus p → satisfiesConditions p)

def statementD : Prop :=
  (∀ p : Point, ¬onLocus p → ¬satisfiesConditions p) ∧
  (∀ p : Point, ¬satisfiesConditions p → ¬onLocus p)

def statementE : Prop :=
  (∀ p : Point, satisfiesConditions p → onLocus p) ∧
  (∀ p : Point, ¬satisfiesConditions p → ¬onLocus p)

-- Theorem stating which methods are correct and which are incorrect
theorem locus_proof_methods_correctness :
  (statementA satisfiesConditions onLocus) ∧
  (¬statementB satisfiesConditions onLocus) ∧
  (¬statementC satisfiesConditions onLocus) ∧
  (statementD satisfiesConditions onLocus) ∧
  (statementE satisfiesConditions onLocus) :=
sorry

end locus_proof_methods_correctness_l638_63866


namespace max_a_value_l638_63819

theorem max_a_value (a b c d : ℕ+) 
  (h1 : a < 3 * b)
  (h2 : b < 3 * c)
  (h3 : c < 4 * d)
  (h4 : b + d = 200) :
  a ≤ 449 ∧ ∃ (a' b' c' d' : ℕ+), 
    a' = 449 ∧ 
    a' < 3 * b' ∧ 
    b' < 3 * c' ∧ 
    c' < 4 * d' ∧ 
    b' + d' = 200 :=
by sorry

end max_a_value_l638_63819


namespace final_number_not_zero_l638_63892

/-- Represents the operation of replacing two numbers with their sum or difference -/
inductive Operation
  | Sum : ℕ → ℕ → Operation
  | Difference : ℕ → ℕ → Operation

/-- The type representing the state of the blackboard -/
def Blackboard := List ℕ

/-- Applies an operation to the blackboard -/
def applyOperation (board : Blackboard) (op : Operation) : Blackboard :=
  match op with
  | Operation.Sum a b => sorry
  | Operation.Difference a b => sorry

/-- Represents a sequence of operations -/
def OperationSequence := List Operation

/-- Applies a sequence of operations to the initial blackboard -/
def applyOperations (initialBoard : Blackboard) (ops : OperationSequence) : Blackboard :=
  ops.foldl applyOperation initialBoard

/-- The initial state of the blackboard -/
def initialBoard : Blackboard := List.range 1974

theorem final_number_not_zero (ops : OperationSequence) :
  (applyOperations initialBoard ops).length = 1 →
  (applyOperations initialBoard ops).head? ≠ some 0 := by
  sorry

end final_number_not_zero_l638_63892


namespace geometric_sequence_fifth_term_l638_63855

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h_pos : ∀ n, a n > 0)
  (h_prod : a 3 * a 7 = 64) :
  a 5 = 8 :=
sorry

end geometric_sequence_fifth_term_l638_63855


namespace consecutive_integers_sum_l638_63872

theorem consecutive_integers_sum (n : ℕ) (h : n > 0) :
  (n * (n + 1) * (n + 2) = 336) → (n + (n + 1) + (n + 2) = 21) :=
by sorry

end consecutive_integers_sum_l638_63872


namespace cricket_overs_played_l638_63871

/-- Proves that the number of overs played initially in a cricket game is 10, given the specified conditions --/
theorem cricket_overs_played (target : ℝ) (initial_rate : ℝ) (required_rate : ℝ) : 
  target = 242 ∧ initial_rate = 3.2 ∧ required_rate = 5.25 →
  ∃ x : ℝ, x = 10 ∧ target - initial_rate * x = required_rate * (50 - x) := by
  sorry

end cricket_overs_played_l638_63871


namespace dress_price_l638_63826

/-- The final price of a dress after discounts and tax -/
def final_price (d : ℝ) : ℝ :=
  let sale_price := d * (1 - 0.25)
  let staff_price := sale_price * (1 - 0.20)
  let coupon_price := staff_price * (1 - 0.10)
  coupon_price * (1 + 0.08)

/-- Theorem stating the final price of the dress -/
theorem dress_price (d : ℝ) :
  final_price d = 0.5832 * d := by
  sorry

end dress_price_l638_63826


namespace inequality_proof_l638_63857

theorem inequality_proof (a b c d : ℝ) 
  (ha : 0 < a ∧ a < 1) 
  (hb : 0 < b ∧ b < 1) 
  (hc : 0 < c ∧ c < 1) 
  (hd : 0 < d ∧ d < 1) : 
  1 + a * b + b * c + c * d + d * a + a * c + b * d > a + b + c + d := by
  sorry

end inequality_proof_l638_63857


namespace rotate90_clockwise_correct_rotation_result_l638_63882

/-- Rotate a point 90 degrees clockwise around the origin -/
def rotate90Clockwise (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, -p.1)

theorem rotate90_clockwise_correct (x y : ℝ) :
  rotate90Clockwise (x, y) = (y, -x) := by sorry

/-- The original point A -/
def A : ℝ × ℝ := (2, 3)

/-- The rotated point B -/
def B : ℝ × ℝ := rotate90Clockwise A

theorem rotation_result :
  B = (3, -2) := by sorry

end rotate90_clockwise_correct_rotation_result_l638_63882


namespace ten_books_left_to_read_l638_63869

/-- The number of books left to read in the 'crazy silly school' series -/
def books_left_to_read (total_books read_books : ℕ) : ℕ :=
  total_books - read_books

/-- Theorem stating that there are 10 books left to read -/
theorem ten_books_left_to_read :
  books_left_to_read 22 12 = 10 := by
  sorry

#eval books_left_to_read 22 12

end ten_books_left_to_read_l638_63869


namespace contrapositive_equivalence_dot_product_not_sufficient_exp_not_periodic_negation_existential_l638_63873

-- 1. Contrapositive
theorem contrapositive_equivalence (p q : ℝ) :
  (p^2 + q^2 = 2 → p + q ≤ 2) ↔ (p + q > 2 → p^2 + q^2 ≠ 2) := by sorry

-- 2. Vector dot product
theorem dot_product_not_sufficient (a b c : ℝ × ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) :
  (a.1 * b.1 + a.2 * b.2 = b.1 * c.1 + b.2 * c.2) → (a = c → False) := by sorry

-- 3. Non-periodicity of exponential function
theorem exp_not_periodic (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ¬∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, a^x = a^(x + T) := by sorry

-- 4. Negation of existential proposition
theorem negation_existential :
  (¬∃ x : ℝ, x^2 - 3*x + 2 ≥ 0) ↔ (∀ x : ℝ, x^2 - 3*x + 2 < 0) := by sorry

end contrapositive_equivalence_dot_product_not_sufficient_exp_not_periodic_negation_existential_l638_63873


namespace valid_three_digit_count_l638_63837

/-- The count of valid three-digit numbers -/
def valid_count : ℕ := 90

/-- The total count of three-digit numbers -/
def total_three_digit : ℕ := 900

/-- The count of three-digit numbers with exactly two different non-adjacent digits -/
def excluded_count : ℕ := 810

/-- Theorem stating that the count of valid three-digit numbers is correct -/
theorem valid_three_digit_count :
  valid_count = total_three_digit - excluded_count :=
by sorry

end valid_three_digit_count_l638_63837


namespace polynomial_multiplication_identity_l638_63825

theorem polynomial_multiplication_identity (x y : ℝ) :
  (3 * x^4 - 7 * y^3) * (9 * x^8 + 21 * x^4 * y^3 + 49 * y^6) = 27 * x^12 - 343 * y^9 := by
  sorry

end polynomial_multiplication_identity_l638_63825


namespace pure_imaginary_ratio_l638_63886

theorem pure_imaginary_ratio (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) 
  (h3 : ∃ (k : ℝ), (2 - 7*I) * (a + b*I) = k*I) : a/b = -7/2 := by
  sorry

end pure_imaginary_ratio_l638_63886


namespace ratio_change_l638_63807

theorem ratio_change (x : ℤ) : 
  (4 * x = 16) →  -- The larger integer is 16, and it's 4 times the smaller integer
  ((x + 12) / (4 * x) = 1) -- The new ratio after adding 12 to the smaller integer is 1:1
:= by sorry

end ratio_change_l638_63807


namespace alloy_mixture_theorem_l638_63808

/-- Represents an alloy with given ratios of gold, silver, and copper -/
structure Alloy where
  gold : ℚ
  silver : ℚ
  copper : ℚ

/-- The total ratio of an alloy -/
def Alloy.total (a : Alloy) : ℚ := a.gold + a.silver + a.copper

/-- Create an alloy from integer ratios -/
def Alloy.fromRatios (g s c : ℕ) : Alloy :=
  let t : ℚ := (g + s + c : ℚ)
  { gold := g / t, silver := s / t, copper := c / t }

theorem alloy_mixture_theorem (x y z : ℚ) :
  let a1 := Alloy.fromRatios 1 3 5
  let a2 := Alloy.fromRatios 3 5 1
  let a3 := Alloy.fromRatios 5 1 3
  let total_mass : ℚ := 351
  let desired_ratio := Alloy.fromRatios 7 9 11
  x = 195 ∧ y = 78 ∧ z = 78 →
  x + y + z = total_mass ∧
  (x * a1.gold + y * a2.gold + z * a3.gold) / total_mass = desired_ratio.gold ∧
  (x * a1.silver + y * a2.silver + z * a3.silver) / total_mass = desired_ratio.silver ∧
  (x * a1.copper + y * a2.copper + z * a3.copper) / total_mass = desired_ratio.copper := by
  sorry

end alloy_mixture_theorem_l638_63808


namespace primes_between_50_and_60_l638_63827

def count_primes_in_range (a b : ℕ) : ℕ :=
  (Finset.range (b - a + 1)).filter (fun i => Nat.Prime (i + a)) |>.card

theorem primes_between_50_and_60 : count_primes_in_range 50 60 = 2 := by
  sorry

end primes_between_50_and_60_l638_63827


namespace quadratic_roots_theorem_l638_63862

theorem quadratic_roots_theorem (b c : ℝ) : 
  ({1, 2} : Set ℝ) = {x | x^2 + b*x + c = 0} → b = -3 ∧ c = 2 := by
  sorry

end quadratic_roots_theorem_l638_63862


namespace probability_of_C_l638_63894

/-- A board game spinner with six regions -/
structure Spinner :=
  (probA : ℚ)
  (probB : ℚ)
  (probC : ℚ)
  (probD : ℚ)
  (probE : ℚ)
  (probF : ℚ)

/-- The conditions of the spinner -/
def spinnerConditions (s : Spinner) : Prop :=
  s.probA = 2/9 ∧
  s.probB = 1/6 ∧
  s.probC = s.probD ∧
  s.probC = s.probE ∧
  s.probF = 2 * s.probC ∧
  s.probA + s.probB + s.probC + s.probD + s.probE + s.probF = 1

/-- The theorem stating the probability of region C -/
theorem probability_of_C (s : Spinner) (h : spinnerConditions s) : s.probC = 11/90 := by
  sorry

end probability_of_C_l638_63894


namespace circle_properties_l638_63801

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 1)^2 = 5

-- Define the points A and B
def point_A : ℝ × ℝ := (-1, 0)
def point_B : ℝ × ℝ := (3, 0)

-- Define the line x - y = 0
def center_line (x y : ℝ) : Prop := x = y

-- Define the line x + 2y + 4 = 0
def distance_line (x y : ℝ) : Prop := x + 2*y + 4 = 0

-- Main theorem
theorem circle_properties :
  -- The circle passes through A and B
  circle_C point_A.1 point_A.2 ∧ circle_C point_B.1 point_B.2 ∧
  -- The center is on the line x - y = 0
  ∃ x y, circle_C x y ∧ center_line x y ∧
  -- Maximum and minimum distances
  (∀ x y, circle_C x y →
    (∃ d_max, d_max = (12/5)*Real.sqrt 5 ∧
      ∀ d, (∃ x' y', distance_line x' y' ∧ d = Real.sqrt ((x - x')^2 + (y - y')^2)) → d ≤ d_max) ∧
    (∃ d_min, d_min = (2/5)*Real.sqrt 5 ∧
      ∀ d, (∃ x' y', distance_line x' y' ∧ d = Real.sqrt ((x - x')^2 + (y - y')^2)) → d ≥ d_min)) :=
by sorry

end circle_properties_l638_63801


namespace geometric_sequence_product_l638_63897

/-- Given a geometric sequence {aₙ} where all terms are positive,
    prove that a₅a₇a₉ = 12 when a₂a₄a₆ = 6 and a₈a₁₀a₁₂ = 24 -/
theorem geometric_sequence_product (a : ℕ → ℝ) :
  (∀ n, a n > 0) →
  a 2 * a 4 * a 6 = 6 →
  a 8 * a 10 * a 12 = 24 →
  a 5 * a 7 * a 9 = 12 := by
  sorry

end geometric_sequence_product_l638_63897


namespace length_of_AB_l638_63803

-- Define the line l: kx + y - 2 = 0
def line_l (k : ℝ) (x y : ℝ) : Prop := k * x + y - 2 = 0

-- Define the circle C: x² + y² - 6x + 2y + 9 = 0
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 2*y + 9 = 0

-- Define that line l is the axis of symmetry for circle C
def is_axis_of_symmetry (k : ℝ) : Prop := 
  ∀ x y : ℝ, line_l k x y → (∃ x' y' : ℝ, circle_C x' y' ∧ 
    ((x - x')^2 + (y - y')^2 = (x' - 3)^2 + (y' + 1)^2))

-- Define point A
def point_A (k : ℝ) : ℝ × ℝ := (0, k)

-- Define that there exists a tangent line from A to circle C
def exists_tangent (k : ℝ) : Prop :=
  ∃ x y : ℝ, circle_C x y ∧ 
    ((x - 0)^2 + (y - k)^2) * ((x - 3)^2 + (y + 1)^2) = 1

-- Theorem statement
theorem length_of_AB (k : ℝ) : 
  is_axis_of_symmetry k → exists_tangent k → 
  ∃ x y : ℝ, circle_C x y ∧ 
    Real.sqrt ((x - 0)^2 + (y - k)^2) = 2 * Real.sqrt 3 :=
sorry

end length_of_AB_l638_63803


namespace triangle_at_most_one_obtuse_l638_63875

-- Define a triangle
structure Triangle where
  angles : Fin 3 → ℝ
  sum_180 : (angles 0) + (angles 1) + (angles 2) = 180
  all_positive : ∀ i, 0 < angles i

-- Define an obtuse angle
def is_obtuse (angle : ℝ) : Prop := 90 < angle

-- Theorem statement
theorem triangle_at_most_one_obtuse (t : Triangle) :
  ¬(∃ i j : Fin 3, i ≠ j ∧ is_obtuse (t.angles i) ∧ is_obtuse (t.angles j)) :=
sorry

end triangle_at_most_one_obtuse_l638_63875


namespace area_ratio_is_one_seventh_l638_63895

/-- Given a triangle XYZ with sides XY, YZ, XZ and points P on XY and Q on XZ,
    this function calculates the ratio of the area of triangle XPQ to the area of quadrilateral PQYZ -/
def areaRatio (XY YZ XZ XP XQ : ℝ) : ℝ :=
  -- Define the ratio calculation here
  sorry

/-- Theorem stating that for the given triangle and points, the area ratio is 1/7 -/
theorem area_ratio_is_one_seventh :
  areaRatio 24 52 60 12 20 = 1 / 7 := by
  sorry

end area_ratio_is_one_seventh_l638_63895


namespace quadratic_equation_roots_l638_63813

theorem quadratic_equation_roots : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  (3 * x₁^2 - 3 * x₁ - 4 = 0) ∧ (3 * x₂^2 - 3 * x₂ - 4 = 0) := by
  sorry

end quadratic_equation_roots_l638_63813


namespace gcd_lcm_product_180_l638_63843

def count_gcd_values (n : ℕ) : Prop :=
  ∃ (S : Finset ℕ),
    (∀ a b : ℕ, (Nat.gcd a b) * (Nat.lcm a b) = n →
      (Nat.gcd a b) ∈ S) ∧
    S.card = 8

theorem gcd_lcm_product_180 :
  count_gcd_values 180 := by
sorry

end gcd_lcm_product_180_l638_63843


namespace right_triangle_sin_A_l638_63858

theorem right_triangle_sin_A (A B C : Real) (h1 : 3 * Real.sin A = 2 * Real.cos A) 
  (h2 : Real.cos B = 0) : Real.sin A = 2 * Real.sqrt 13 / 13 := by
  sorry

end right_triangle_sin_A_l638_63858


namespace triangle_side_constraint_l638_63822

theorem triangle_side_constraint (a : ℝ) : 
  (0 < a) → (0 < 2) → (0 < 6) → 
  (2 + 6 > a) → (6 + a > 2) → (2 + a > 6) → 
  (4 < a ∧ a < 8) :=
by sorry

end triangle_side_constraint_l638_63822


namespace consecutive_odd_numbers_multiple_l638_63800

theorem consecutive_odd_numbers_multiple (k : ℕ) : 
  let a := 7
  let b := a + 2
  let c := b + 2
  k * a = 3 * c + (2 * b + 5) →
  k = 8 := by
sorry

end consecutive_odd_numbers_multiple_l638_63800


namespace cos_24_cos_36_minus_sin_24_sin_36_l638_63816

theorem cos_24_cos_36_minus_sin_24_sin_36 :
  Real.cos (24 * π / 180) * Real.cos (36 * π / 180) - 
  Real.sin (24 * π / 180) * Real.sin (36 * π / 180) = 1 / 2 := by
  sorry

end cos_24_cos_36_minus_sin_24_sin_36_l638_63816


namespace hundredth_group_sum_divided_by_100_l638_63864

/-- The sum of the first n natural numbers -/
def sum_of_naturals (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The last number in the nth group -/
def last_number (n : ℕ) : ℕ := 2 * sum_of_naturals n

/-- The first number in the nth group -/
def first_number (n : ℕ) : ℕ := last_number (n - 1) - 2 * (n - 1)

/-- The sum of numbers in the nth group -/
def group_sum (n : ℕ) : ℕ := n * (first_number n + last_number n) / 2

theorem hundredth_group_sum_divided_by_100 :
  group_sum 100 / 100 = 10001 := by sorry

end hundredth_group_sum_divided_by_100_l638_63864


namespace system_solution_l638_63848

theorem system_solution (a b c x y z : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (x * y + y * z + z * x = a^2 - x^2) ∧ 
  (x * y + y * z + z * x = b^2 - y^2) ∧ 
  (x * y + y * z + z * x = c^2 - z^2) →
  ((x = (|a*b/c| + |a*c/b| - |b*c/a|) / 2 ∧
    y = (|a*b/c| - |a*c/b| + |b*c/a|) / 2 ∧
    z = (-|a*b/c| + |a*c/b| + |b*c/a|) / 2) ∨
   (x = -(|a*b/c| + |a*c/b| - |b*c/a|) / 2 ∧
    y = -(|a*b/c| - |a*c/b| + |b*c/a|) / 2 ∧
    z = -(-|a*b/c| + |a*c/b| + |b*c/a|) / 2)) :=
by sorry

end system_solution_l638_63848


namespace slips_with_three_l638_63840

theorem slips_with_three (total_slips : ℕ) (value_a value_b : ℕ) (expected_value : ℚ) : 
  total_slips = 15 →
  value_a = 3 →
  value_b = 8 →
  expected_value = 5 →
  ∃ (slips_with_a : ℕ),
    slips_with_a ≤ total_slips ∧
    (slips_with_a : ℚ) / total_slips * value_a + 
    ((total_slips - slips_with_a) : ℚ) / total_slips * value_b = expected_value ∧
    slips_with_a = 9 := by
sorry

end slips_with_three_l638_63840


namespace flower_combination_l638_63805

theorem flower_combination : Nat.choose 10 6 = 210 := by
  sorry

end flower_combination_l638_63805


namespace sum_not_zero_l638_63817

theorem sum_not_zero (a b c d : ℝ) 
  (eq1 : a * b * c - d = 1)
  (eq2 : b * c * d - a = 2)
  (eq3 : c * d * a - b = 3)
  (eq4 : d * a * b - c = -6) : 
  a + b + c + d ≠ 0 := by
  sorry

end sum_not_zero_l638_63817


namespace greatest_b_value_l638_63828

theorem greatest_b_value (b : ℝ) : 
  (∀ x : ℝ, x^2 - 12*x + 35 ≤ 0 → x ≤ 7) ∧ 
  (7^2 - 12*7 + 35 ≤ 0) :=
by sorry

end greatest_b_value_l638_63828


namespace ferris_wheel_rides_correct_l638_63887

/-- The number of times Billy rode the ferris wheel -/
def ferris_wheel_rides : ℕ := 7

/-- The number of times Billy rode the bumper cars -/
def bumper_car_rides : ℕ := 3

/-- The cost of each ride in tickets -/
def cost_per_ride : ℕ := 5

/-- The total number of tickets Billy used -/
def total_tickets : ℕ := 50

/-- Theorem stating that the number of ferris wheel rides is correct -/
theorem ferris_wheel_rides_correct : 
  ferris_wheel_rides * cost_per_ride + bumper_car_rides * cost_per_ride = total_tickets :=
by sorry

end ferris_wheel_rides_correct_l638_63887


namespace box_volume_increase_l638_63810

theorem box_volume_increase (l w h : ℝ) 
  (volume : l * w * h = 5000)
  (surface_area : 2 * (l * w + w * h + h * l) = 1800)
  (edge_sum : 4 * (l + w + h) = 240) :
  (l + 2) * (w + 2) * (h + 2) = 7048 := by
sorry

end box_volume_increase_l638_63810


namespace sequence_general_term_l638_63814

theorem sequence_general_term (n : ℕ) (a : ℕ → ℕ) (S : ℕ → ℕ) :
  (∀ k, S k = k^2 + k) →
  (a 1 = S 1) →
  (∀ k ≥ 2, a k = S k - S (k - 1)) →
  ∀ k, a k = 2 * k :=
sorry

end sequence_general_term_l638_63814


namespace no_perfect_squares_in_all_ones_sequence_l638_63877

/-- Represents a number in the sequence 11, 111, 1111, ... -/
def allOnesNumber (n : ℕ) : ℕ :=
  (10^n - 1) / 9

/-- Predicate to check if a natural number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop :=
  ∃ k : ℕ, k^2 = n

/-- Theorem: There are no perfect squares in the sequence of numbers
    consisting of only the digit 1, starting from 11 -/
theorem no_perfect_squares_in_all_ones_sequence :
  ∀ n : ℕ, n ≥ 2 → ¬ isPerfectSquare (allOnesNumber n) :=
sorry

end no_perfect_squares_in_all_ones_sequence_l638_63877


namespace special_gp_common_ratio_l638_63833

/-- A geometric progression where each term, starting from the third, 
    is equal to the sum of the two preceding terms. -/
def SpecialGeometricProgression (u : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ), ∀ (n : ℕ),
    u (n + 1) = u n * q ∧ 
    u (n + 2) = u (n + 1) + u n

/-- The common ratio of a special geometric progression 
    is either (1 + √5) / 2 or (1 - √5) / 2. -/
theorem special_gp_common_ratio 
  (u : ℕ → ℝ) (h : SpecialGeometricProgression u) : 
  ∃ (q : ℝ), (∀ (n : ℕ), u (n + 1) = u n * q) ∧ 
    (q = (1 + Real.sqrt 5) / 2 ∨ q = (1 - Real.sqrt 5) / 2) := by
  sorry

end special_gp_common_ratio_l638_63833


namespace men_on_first_road_calculation_l638_63853

/-- The number of men who worked on the first road -/
def men_on_first_road : ℕ := 30

/-- The length of the first road in kilometers -/
def first_road_length : ℕ := 1

/-- The number of days spent working on the first road -/
def days_on_first_road : ℕ := 12

/-- The number of hours worked per day on the first road -/
def hours_per_day_first_road : ℕ := 8

/-- The number of men working on the second road -/
def men_on_second_road : ℕ := 20

/-- The number of days spent working on the second road -/
def days_on_second_road : ℕ := 32

/-- The number of hours worked per day on the second road -/
def hours_per_day_second_road : ℕ := 9

/-- The length of the second road in kilometers -/
def second_road_length : ℕ := 2

theorem men_on_first_road_calculation :
  men_on_first_road * days_on_first_road * hours_per_day_first_road =
  (men_on_second_road * days_on_second_road * hours_per_day_second_road) / 2 :=
by sorry

end men_on_first_road_calculation_l638_63853


namespace D_is_empty_l638_63836

-- Define the set D
def D : Set ℝ := {x : ℝ | x^2 + 2 = 0}

-- Theorem stating that D is an empty set
theorem D_is_empty : D = ∅ := by sorry

end D_is_empty_l638_63836


namespace student_allowance_proof_l638_63844

/-- The student's weekly allowance in dollars -/
def weekly_allowance : ℝ := 4.50

theorem student_allowance_proof :
  ∃ (arcade_spent toy_store_spent : ℝ),
    arcade_spent = (3/5) * weekly_allowance ∧
    toy_store_spent = (1/3) * (weekly_allowance - arcade_spent) ∧
    weekly_allowance - arcade_spent - toy_store_spent = 1.20 :=
by sorry

end student_allowance_proof_l638_63844


namespace x_in_terms_of_y_and_z_l638_63854

theorem x_in_terms_of_y_and_z (x y z : ℝ) :
  1 / (x + y) + 1 / (x - y) = z / (x - y) → x = z / 2 := by
  sorry

end x_in_terms_of_y_and_z_l638_63854


namespace tangent_line_sum_l638_63884

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem tangent_line_sum (h : ∀ y, y = f 2 → y = 2 * 2 + 3) : f 2 + (deriv f) 2 = 9 := by
  sorry

end tangent_line_sum_l638_63884


namespace complex_equation_solution_l638_63829

variable (i : ℂ)
variable (z : ℂ)

theorem complex_equation_solution (hi : i * i = -1) (hz : (1 + i) / z = 1 - i) : z = i := by
  sorry

end complex_equation_solution_l638_63829


namespace base_number_proof_l638_63823

theorem base_number_proof (x y : ℝ) (h1 : x ^ y = 3 ^ 16) (h2 : y = 8) : x = 9 := by
  sorry

end base_number_proof_l638_63823


namespace bianca_candy_problem_l638_63889

/-- Bianca's Halloween candy problem -/
theorem bianca_candy_problem (initial_candy : ℕ) (piles : ℕ) (pieces_per_pile : ℕ) 
  (h1 : initial_candy = 32)
  (h2 : piles = 4)
  (h3 : pieces_per_pile = 5) :
  initial_candy - (piles * pieces_per_pile) = 12 := by
  sorry

end bianca_candy_problem_l638_63889
