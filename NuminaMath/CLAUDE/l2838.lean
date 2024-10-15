import Mathlib

namespace NUMINAMATH_CALUDE_unique_perfect_square_solution_l2838_283873

theorem unique_perfect_square_solution (n : ℤ) : 
  (∃ k : ℤ, n^4 + 6*n^3 + 11*n^2 + 3*n + 31 = k^2) ↔ n = 10 := by
sorry

end NUMINAMATH_CALUDE_unique_perfect_square_solution_l2838_283873


namespace NUMINAMATH_CALUDE_radian_measure_of_15_degrees_l2838_283870

theorem radian_measure_of_15_degrees :
  let degree_to_radian (d : ℝ) := d * (Real.pi / 180)
  degree_to_radian 15 = Real.pi / 12 := by
  sorry

end NUMINAMATH_CALUDE_radian_measure_of_15_degrees_l2838_283870


namespace NUMINAMATH_CALUDE_rectangle_area_increase_l2838_283829

theorem rectangle_area_increase : 
  let original_length : ℝ := 40
  let original_width : ℝ := 20
  let length_decrease : ℝ := 5
  let width_increase : ℝ := 5
  let new_length : ℝ := original_length - length_decrease
  let new_width : ℝ := original_width + width_increase
  let original_area : ℝ := original_length * original_width
  let new_area : ℝ := new_length * new_width
  new_area - original_area = 75
  := by sorry

end NUMINAMATH_CALUDE_rectangle_area_increase_l2838_283829


namespace NUMINAMATH_CALUDE_nonagon_side_length_l2838_283841

/-- The length of one side of a regular nonagon with circumference 171 cm is 19 cm. -/
theorem nonagon_side_length : 
  ∀ (circumference side_length : ℝ),
  circumference = 171 →
  side_length * 9 = circumference →
  side_length = 19 := by
sorry

end NUMINAMATH_CALUDE_nonagon_side_length_l2838_283841


namespace NUMINAMATH_CALUDE_chord_division_ratio_l2838_283863

/-- Given a circle with radius 11 and a chord of length 18 that intersects
    a diameter at a point 7 units from the center, prove that this point
    divides the chord in a ratio of either 2:1 or 1:2. -/
theorem chord_division_ratio (R : ℝ) (chord_length : ℝ) (center_to_intersection : ℝ)
    (h1 : R = 11)
    (h2 : chord_length = 18)
    (h3 : center_to_intersection = 7) :
    ∃ (x y : ℝ), (x + y = chord_length ∧ 
                 ((x / y = 2 ∧ y / x = 1/2) ∨ 
                  (x / y = 1/2 ∧ y / x = 2))) :=
by sorry

end NUMINAMATH_CALUDE_chord_division_ratio_l2838_283863


namespace NUMINAMATH_CALUDE_x_twelve_equals_one_l2838_283851

theorem x_twelve_equals_one (x : ℝ) (h : x + 1/x = 2) : x^12 = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_twelve_equals_one_l2838_283851


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2838_283801

theorem quadratic_inequality_solution (x : ℝ) :
  x^2 - 50*x + 576 ≤ 16 ↔ 20 ≤ x ∧ x ≤ 28 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2838_283801


namespace NUMINAMATH_CALUDE_equation_solution_range_l2838_283828

theorem equation_solution_range (b : ℝ) : 
  (∀ x : ℝ, x = -2 → x^2 - b*x - 5 = 5) →
  (∀ x : ℝ, x = -1 → x^2 - b*x - 5 = -1) →
  (∀ x : ℝ, x = 4 → x^2 - b*x - 5 = -1) →
  (∀ x : ℝ, x = 5 → x^2 - b*x - 5 = 5) →
  ∃ x y : ℝ, 
    (-2 < x ∧ x < -1 ∧ x^2 - b*x - 5 = 0) ∧
    (4 < y ∧ y < 5 ∧ y^2 - b*y - 5 = 0) ∧
    (∀ z : ℝ, z^2 - b*z - 5 = 0 → ((-2 < z ∧ z < -1) ∨ (4 < z ∧ z < 5))) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_range_l2838_283828


namespace NUMINAMATH_CALUDE_tyrones_money_equals_thirteen_dollars_l2838_283807

/-- The value of Tyrone's money in cents -/
def tyrones_money : ℕ :=
  2 * 100 +  -- Two $1 bills
  5 * 100 +  -- One $5 bill
  13 * 25 +  -- 13 quarters
  20 * 10 +  -- 20 dimes
  8 * 5 +    -- 8 nickels
  35 * 1     -- 35 pennies

/-- Theorem stating that Tyrone's money equals $13 -/
theorem tyrones_money_equals_thirteen_dollars :
  tyrones_money = 13 * 100 := by
  sorry

end NUMINAMATH_CALUDE_tyrones_money_equals_thirteen_dollars_l2838_283807


namespace NUMINAMATH_CALUDE_min_product_of_prime_sum_l2838_283823

theorem min_product_of_prime_sum (m n p : ℕ) : 
  Prime m → Prime n → Prime p → m ≠ n → n ≠ p → m ≠ p → m + n = p → 
  (∀ m' n' p' : ℕ, Prime m' → Prime n' → Prime p' → m' ≠ n' → n' ≠ p' → m' ≠ p' → m' + n' = p' → m' * n' * p' ≥ m * n * p) →
  m * n * p = 30 := by
sorry

end NUMINAMATH_CALUDE_min_product_of_prime_sum_l2838_283823


namespace NUMINAMATH_CALUDE_sin_alpha_minus_pi_sixth_l2838_283827

theorem sin_alpha_minus_pi_sixth (α : Real) 
  (h : Real.cos (α - π/3) - Real.cos α = 1/3) : 
  Real.sin (α - π/6) = 1/3 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_minus_pi_sixth_l2838_283827


namespace NUMINAMATH_CALUDE_triangle_trig_identity_l2838_283839

theorem triangle_trig_identity (A B C : ℝ) (hABC : A + B + C = π) 
  (hAC : 2 = Real.sqrt ((B - C)^2 + 4 * (Real.sin (A/2))^2)) 
  (hBC : 3 = Real.sqrt ((A - C)^2 + 4 * (Real.sin (B/2))^2))
  (hcosA : Real.cos A = -4/5) :
  Real.sin (2*B + π/6) = (17 + 12 * Real.sqrt 7) / 25 := by
  sorry

end NUMINAMATH_CALUDE_triangle_trig_identity_l2838_283839


namespace NUMINAMATH_CALUDE_ceiling_equation_solution_l2838_283854

theorem ceiling_equation_solution :
  ∃! b : ℝ, b + ⌈b⌉ = 14.7 ∧ b = 7.2 := by sorry

end NUMINAMATH_CALUDE_ceiling_equation_solution_l2838_283854


namespace NUMINAMATH_CALUDE_can_determine_coin_type_l2838_283821

/-- Represents the outcome of weighing two groups of coins -/
inductive WeighingResult
  | Even
  | Odd

/-- Represents the type of a coin -/
inductive CoinType
  | Genuine
  | Counterfeit

/-- Function to weigh two groups of coins -/
def weigh (group1 : Finset Nat) (group2 : Finset Nat) : WeighingResult :=
  sorry

theorem can_determine_coin_type 
  (total_coins : Nat)
  (counterfeit_coins : Nat)
  (weight_difference : Nat)
  (h1 : total_coins = 101)
  (h2 : counterfeit_coins = 50)
  (h3 : weight_difference = 1)
  : ∃ (f : Finset Nat → Finset Nat → WeighingResult → CoinType), 
    ∀ (selected_coin : Nat) (group1 group2 : Finset Nat),
    selected_coin ∉ group1 ∧ selected_coin ∉ group2 →
    group1.card = 50 ∧ group2.card = 50 →
    f group1 group2 (weigh group1 group2) = 
      if selected_coin ≤ counterfeit_coins then CoinType.Counterfeit else CoinType.Genuine :=
sorry

end NUMINAMATH_CALUDE_can_determine_coin_type_l2838_283821


namespace NUMINAMATH_CALUDE_probability_specific_coin_sequence_l2838_283898

/-- The probability of getting a specific sequence of coin flips -/
def probability_specific_sequence (n : ℕ) (p : ℚ) : ℚ :=
  p^n

/-- The number of coin flips -/
def num_flips : ℕ := 10

/-- The probability of getting tails on a single flip -/
def prob_tails : ℚ := 1/2

/-- Theorem: The probability of getting the sequence TTT HHHH THT in 10 coin flips -/
theorem probability_specific_coin_sequence :
  probability_specific_sequence num_flips prob_tails = 1/1024 := by
  sorry

end NUMINAMATH_CALUDE_probability_specific_coin_sequence_l2838_283898


namespace NUMINAMATH_CALUDE_simplify_fraction_l2838_283850

theorem simplify_fraction (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) :
  2 / (x^2 - 1) - 1 / (x - 1) = -1 / (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2838_283850


namespace NUMINAMATH_CALUDE_cone_volume_proof_l2838_283847

theorem cone_volume_proof (a b c r h : ℝ) : 
  a = 3 → b = 4 → c^2 = a^2 + b^2 → 
  2 * r = c → h^2 + r^2 = 3^2 →
  (1/3) * π * r^2 * h = (25 * π * Real.sqrt 11) / 24 := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_proof_l2838_283847


namespace NUMINAMATH_CALUDE_digit_rearrangement_divisibility_l2838_283887

def is_digit_rearrangement (n m : ℕ) : Prop :=
  ∃ (digits_n digits_m : List ℕ), 
    digits_n.length > 0 ∧
    digits_m.length > 0 ∧
    digits_n.sum = digits_m.sum ∧
    n = digits_n.foldl (λ acc d => acc * 10 + d) 0 ∧
    m = digits_m.foldl (λ acc d => acc * 10 + d) 0

def satisfies_property (d : ℕ) : Prop :=
  d > 0 ∧ ∀ n m : ℕ, n > 0 → is_digit_rearrangement n m → (d ∣ n → d ∣ m)

theorem digit_rearrangement_divisibility :
  {d : ℕ | satisfies_property d} = {1, 3, 9} := by sorry

end NUMINAMATH_CALUDE_digit_rearrangement_divisibility_l2838_283887


namespace NUMINAMATH_CALUDE_last_remaining_number_l2838_283815

/-- The function that determines the next position of a number after one round of erasure -/
def nextPosition (n : ℕ) : ℕ :=
  if n % 3 = 0 then 2 * (n / 3)
  else if n % 3 = 2 then 2 * (n / 3) + 1
  else 0  -- This case (n % 3 = 1) corresponds to erased numbers

/-- The function that determines the original position given a final position -/
def originalPosition (finalPos : ℕ) : ℕ :=
  if finalPos = 1 then 1458 else 0  -- We only care about the winning position

/-- The theorem stating that 1458 is the last remaining number -/
theorem last_remaining_number :
  ∃ (n : ℕ), n ≤ 2002 ∧ 
  (∀ (m : ℕ), m ≤ 2002 → m ≠ n → 
    ∃ (k : ℕ), originalPosition m = 3 * k + 1 ∨ 
    ∃ (j : ℕ), nextPosition (originalPosition m) = originalPosition j ∧ j < m) ∧
  originalPosition n = n ∧ n = 1458 :=
sorry

end NUMINAMATH_CALUDE_last_remaining_number_l2838_283815


namespace NUMINAMATH_CALUDE_wind_velocity_problem_l2838_283809

/-- Represents the relationship between pressure, area, and wind velocity -/
def pressure_relation (k : ℝ) (A V : ℝ) : ℝ := k * A * V^3

theorem wind_velocity_problem (k : ℝ) :
  let A₁ : ℝ := 1
  let V₁ : ℝ := 10
  let P₁ : ℝ := 1
  let A₂ : ℝ := 1
  let P₂ : ℝ := 64
  pressure_relation k A₁ V₁ = P₁ →
  pressure_relation k A₂ 40 = P₂ :=
by
  sorry

#check wind_velocity_problem

end NUMINAMATH_CALUDE_wind_velocity_problem_l2838_283809


namespace NUMINAMATH_CALUDE_sqrt_one_minus_x_domain_l2838_283846

theorem sqrt_one_minus_x_domain : ∀ x : ℝ, 
  (x ≤ 1 ↔ ∃ y : ℝ, y ^ 2 = 1 - x) ∧
  (x = 2 → ¬∃ y : ℝ, y ^ 2 = 1 - x) :=
sorry

end NUMINAMATH_CALUDE_sqrt_one_minus_x_domain_l2838_283846


namespace NUMINAMATH_CALUDE_log_equation_solution_l2838_283859

theorem log_equation_solution (x : ℝ) :
  (Real.log x / Real.log 4) + (Real.log (1/6) / Real.log 4) = 1/2 → x = 12 := by
sorry

end NUMINAMATH_CALUDE_log_equation_solution_l2838_283859


namespace NUMINAMATH_CALUDE_dannys_bottle_caps_l2838_283824

theorem dannys_bottle_caps (park_caps : ℕ) (park_wrappers : ℕ) (collection_wrappers : ℕ) :
  park_caps = 58 →
  park_wrappers = 25 →
  collection_wrappers = 11 →
  ∃ (collection_caps : ℕ), collection_caps = collection_wrappers + 1 ∧ collection_caps = 12 :=
by sorry

end NUMINAMATH_CALUDE_dannys_bottle_caps_l2838_283824


namespace NUMINAMATH_CALUDE_pet_owners_problem_l2838_283817

theorem pet_owners_problem (total_pet_owners : ℕ) (only_dogs : ℕ) (only_cats : ℕ) (cats_dogs_snakes : ℕ) (total_snakes : ℕ)
  (h1 : total_pet_owners = 79)
  (h2 : only_dogs = 15)
  (h3 : only_cats = 10)
  (h4 : cats_dogs_snakes = 3)
  (h5 : total_snakes = 49) :
  total_pet_owners - only_dogs - only_cats - cats_dogs_snakes - (total_snakes - cats_dogs_snakes) = 5 :=
by sorry

end NUMINAMATH_CALUDE_pet_owners_problem_l2838_283817


namespace NUMINAMATH_CALUDE_sara_frosting_cans_l2838_283862

/-- The number of cans of frosting needed to frost the remaining cakes after Sara's baking and Carol's eating -/
def frosting_cans_needed (cakes_per_day : ℕ) (days : ℕ) (cakes_eaten : ℕ) (frosting_cans_per_cake : ℕ) : ℕ :=
  ((cakes_per_day * days - cakes_eaten) * frosting_cans_per_cake)

/-- Theorem stating the number of frosting cans needed in Sara's specific scenario -/
theorem sara_frosting_cans : frosting_cans_needed 10 5 12 2 = 76 := by
  sorry

end NUMINAMATH_CALUDE_sara_frosting_cans_l2838_283862


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l2838_283822

theorem square_area_from_diagonal : 
  ∀ (d s A : ℝ), 
  d = 8 * Real.sqrt 2 →  -- diagonal length
  d = s * Real.sqrt 2 →  -- relationship between diagonal and side
  A = s^2 →             -- area formula
  A = 64 := by           
sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l2838_283822


namespace NUMINAMATH_CALUDE_isometric_figure_area_l2838_283877

/-- A horizontally placed figure with an isometric view -/
structure IsometricFigure where
  /-- The isometric view is an isosceles right triangle -/
  isIsoscelesRightTriangle : Prop
  /-- The legs of the isometric view triangle have length 1 -/
  legLength : ℝ
  /-- The area of the isometric view -/
  isometricArea : ℝ
  /-- The area of the original plane figure -/
  originalArea : ℝ

/-- 
  If a horizontally placed figure has an isometric view that is an isosceles right triangle 
  with legs of length 1, then the area of the original plane figure is √2.
-/
theorem isometric_figure_area 
  (fig : IsometricFigure) 
  (h1 : fig.isIsoscelesRightTriangle) 
  (h2 : fig.legLength = 1) 
  (h3 : fig.isometricArea = 1 / 2) : 
  fig.originalArea = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_isometric_figure_area_l2838_283877


namespace NUMINAMATH_CALUDE_somu_father_age_ratio_l2838_283852

/-- Represents the ages of Somu and his father -/
structure Ages where
  somu : ℕ
  father : ℕ

/-- The condition that Somu's age 10 years ago was one-fifth of his father's age 10 years ago -/
def age_condition (ages : Ages) : Prop :=
  ages.somu - 10 = (ages.father - 10) / 5

/-- The theorem stating that given Somu's present age is 20 and the age condition,
    the ratio of Somu's present age to his father's present age is 1:3 -/
theorem somu_father_age_ratio (ages : Ages) 
    (h1 : ages.somu = 20) 
    (h2 : age_condition ages) : 
    (ages.somu : ℚ) / ages.father = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_somu_father_age_ratio_l2838_283852


namespace NUMINAMATH_CALUDE_pokemon_cards_total_l2838_283869

theorem pokemon_cards_total (jason_left : ℕ) (jason_gave : ℕ) (lisa_left : ℕ) (lisa_gave : ℕ) :
  jason_left = 4 → jason_gave = 9 → lisa_left = 7 → lisa_gave = 15 →
  (jason_left + jason_gave) + (lisa_left + lisa_gave) = 35 :=
by sorry

end NUMINAMATH_CALUDE_pokemon_cards_total_l2838_283869


namespace NUMINAMATH_CALUDE_not_q_is_false_l2838_283895

theorem not_q_is_false (p q : Prop) (hp : ¬p) (hq : q) : ¬(¬q) := by
  sorry

end NUMINAMATH_CALUDE_not_q_is_false_l2838_283895


namespace NUMINAMATH_CALUDE_third_draw_probability_l2838_283867

/-- Represents the number of balls of each color in the box -/
structure BallCount where
  white : ℕ
  black : ℕ

/-- Calculates the probability of drawing a white ball -/
def probWhite (balls : BallCount) : ℚ :=
  balls.white / (balls.white + balls.black)

theorem third_draw_probability :
  let initial := BallCount.mk 8 7
  let after_removal := BallCount.mk (initial.white - 1) (initial.black - 1)
  probWhite after_removal = 7 / 13 := by
  sorry

end NUMINAMATH_CALUDE_third_draw_probability_l2838_283867


namespace NUMINAMATH_CALUDE_parabola_coefficient_l2838_283868

/-- Given a parabola y = ax^2 + bx + c with vertex (q, 2q) and y-intercept (0, -2q), where q ≠ 0, 
    the value of b is 8/q. -/
theorem parabola_coefficient (a b c q : ℝ) (hq : q ≠ 0) : 
  (∀ x y : ℝ, y = a * x^2 + b * x + c) →
  (2 * q = a * q^2 + b * q + c) →
  (-2 * q = c) →
  b = 8 / q := by
sorry

end NUMINAMATH_CALUDE_parabola_coefficient_l2838_283868


namespace NUMINAMATH_CALUDE_arithmetic_geometric_condition_l2838_283813

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def geometric_sequence (a b c : ℝ) : Prop :=
  b * b = a * c

theorem arithmetic_geometric_condition (a : ℕ → ℝ) (d : ℝ) :
  arithmetic_sequence a d ∧ a 1 = 2 →
  (d = 4 → geometric_sequence (a 1) (a 2) (a 5)) ∧
  ¬(geometric_sequence (a 1) (a 2) (a 5) → d = 4) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_condition_l2838_283813


namespace NUMINAMATH_CALUDE_jog_time_proportional_l2838_283892

/-- Given a constant jogging pace, prove that if 3 miles takes 30 minutes, then 1.5 miles takes 15 minutes. -/
theorem jog_time_proportional (pace : ℝ → ℝ) (h_constant : ∀ x y, pace x = pace y) :
  pace 3 = 30 → pace 1.5 = 15 := by
  sorry

end NUMINAMATH_CALUDE_jog_time_proportional_l2838_283892


namespace NUMINAMATH_CALUDE_ralphs_cards_l2838_283818

/-- Given Ralph's initial and additional cards, prove the total number of cards. -/
theorem ralphs_cards (initial_cards additional_cards : ℕ) 
  (h1 : initial_cards = 4)
  (h2 : additional_cards = 8) :
  initial_cards + additional_cards = 12 := by
  sorry

end NUMINAMATH_CALUDE_ralphs_cards_l2838_283818


namespace NUMINAMATH_CALUDE_cos_pi_third_minus_2theta_l2838_283874

theorem cos_pi_third_minus_2theta (θ : ℝ) 
  (h : Real.sin (θ - π / 6) = Real.sqrt 3 / 3) : 
  Real.cos (π / 3 - 2 * θ) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_third_minus_2theta_l2838_283874


namespace NUMINAMATH_CALUDE_train_crossing_time_l2838_283893

/-- The time it takes for a train to cross a pole -/
theorem train_crossing_time (train_speed : ℝ) (train_length : ℝ) : 
  train_speed = 270 →
  train_length = 375.03 →
  (train_length / (train_speed * 1000 / 3600)) = 5.0004 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l2838_283893


namespace NUMINAMATH_CALUDE_cubic_roots_sum_of_squares_l2838_283884

theorem cubic_roots_sum_of_squares (a b c : ℝ) : 
  (∀ x : ℝ, x^3 - 12*x^2 + 47*x - 30 = 0 ↔ (x = a ∨ x = b ∨ x = c)) →
  a^2 + b^2 + c^2 = 50 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_of_squares_l2838_283884


namespace NUMINAMATH_CALUDE_min_value_sum_of_squares_l2838_283860

theorem min_value_sum_of_squares (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : x + y + z = 9) :
  (x^2 + y^2) / (x + y) + (x^2 + z^2) / (x + z) + (y^2 + z^2) / (y + z) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_of_squares_l2838_283860


namespace NUMINAMATH_CALUDE_first_group_men_count_l2838_283814

/-- Represents the number of men in the first group -/
def first_group_men : ℕ := 30

/-- Represents the number of days worked by the first group -/
def first_group_days : ℕ := 12

/-- Represents the number of hours worked per day by the first group -/
def first_group_hours_per_day : ℕ := 8

/-- Represents the length of road (in km) asphalted by the first group -/
def first_group_road_length : ℕ := 1

/-- Represents the number of men in the second group -/
def second_group_men : ℕ := 20

/-- Represents the number of days worked by the second group -/
def second_group_days : ℝ := 19.2

/-- Represents the number of hours worked per day by the second group -/
def second_group_hours_per_day : ℕ := 15

/-- Represents the length of road (in km) asphalted by the second group -/
def second_group_road_length : ℕ := 2

/-- Theorem stating that the number of men in the first group is 30 -/
theorem first_group_men_count : 
  first_group_men * first_group_days * first_group_hours_per_day * second_group_road_length = 
  second_group_men * second_group_days * second_group_hours_per_day * first_group_road_length :=
by sorry

end NUMINAMATH_CALUDE_first_group_men_count_l2838_283814


namespace NUMINAMATH_CALUDE_x_negative_y_positive_l2838_283843

theorem x_negative_y_positive (x y : ℝ) 
  (h1 : 2 * x - y > 3 * x) 
  (h2 : x + 2 * y < 2 * y) : 
  x < 0 ∧ y > 0 := by
  sorry

end NUMINAMATH_CALUDE_x_negative_y_positive_l2838_283843


namespace NUMINAMATH_CALUDE_min_integer_solution_2x_minus_1_geq_5_l2838_283832

theorem min_integer_solution_2x_minus_1_geq_5 :
  ∀ x : ℤ, (2 * x - 1 ≥ 5) → x ≥ 3 ∧ ∀ y : ℤ, (2 * y - 1 ≥ 5) → y ≥ x :=
by sorry

end NUMINAMATH_CALUDE_min_integer_solution_2x_minus_1_geq_5_l2838_283832


namespace NUMINAMATH_CALUDE_max_angle_at_tangent_points_l2838_283826

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- A point in a 2D plane -/
def Point := ℝ × ℝ

/-- Checks if a point is strictly inside a circle -/
def is_inside_circle (p : Point) (c : Circle) : Prop :=
  Real.sqrt ((p.1 - c.center.1)^2 + (p.2 - c.center.2)^2) < c.radius

/-- Checks if a point is on a circle -/
def is_on_circle (p : Point) (c : Circle) : Prop :=
  Real.sqrt ((p.1 - c.center.1)^2 + (p.2 - c.center.2)^2) = c.radius

/-- Calculates the angle ABC given three points A, B, and C -/
noncomputable def angle (A B C : Point) : ℝ := sorry

/-- Defines the tangent points of circles passing through two points and tangent to a given circle -/
noncomputable def tangent_points (A B : Point) (Ω : Circle) : Point × Point := sorry

theorem max_angle_at_tangent_points (Ω : Circle) (A B : Point) :
  is_inside_circle A Ω →
  is_inside_circle B Ω →
  A ≠ B →
  let (C₁, C₂) := tangent_points A B Ω
  ∀ C : Point, is_on_circle C Ω →
    angle A C B ≤ max (angle A C₁ B) (angle A C₂ B) :=
by sorry

end NUMINAMATH_CALUDE_max_angle_at_tangent_points_l2838_283826


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l2838_283890

theorem quadratic_one_solution (n : ℝ) : 
  (∃! x : ℝ, 9 * x^2 + n * x + 36 = 0) ↔ n = 36 ∨ n = -36 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l2838_283890


namespace NUMINAMATH_CALUDE_van_speed_for_longer_time_l2838_283861

/-- Given a van that travels 450 km in 5 hours, this theorem proves the speed
    required to cover the same distance in 3/2 of the original time. -/
theorem van_speed_for_longer_time (distance : ℝ) (initial_time : ℝ) (time_factor : ℝ) :
  distance = 450 ∧ initial_time = 5 ∧ time_factor = 3/2 →
  distance / (initial_time * time_factor) = 60 := by
  sorry

end NUMINAMATH_CALUDE_van_speed_for_longer_time_l2838_283861


namespace NUMINAMATH_CALUDE_distinct_painting_methods_is_catalan_l2838_283880

/-- Represents a ball with a number and color -/
structure Ball where
  number : Nat
  color : Nat

/-- Represents a painting method for n balls -/
def PaintingMethod (n : Nat) := Fin n → Ball

/-- Checks if two painting methods are distinct -/
def is_distinct (n : Nat) (m1 m2 : PaintingMethod n) : Prop :=
  ∃ i : Fin n, (m1 i).color ≠ (m2 i).color

/-- The number of distinct painting methods for n balls -/
def distinct_painting_methods (n : Nat) : Nat :=
  (Nat.choose (2 * n - 2) (n - 1)) / n

/-- Theorem: The number of distinct painting methods is the (n-1)th Catalan number -/
theorem distinct_painting_methods_is_catalan (n : Nat) :
  distinct_painting_methods n = (Nat.choose (2 * n - 2) (n - 1)) / n :=
by sorry

end NUMINAMATH_CALUDE_distinct_painting_methods_is_catalan_l2838_283880


namespace NUMINAMATH_CALUDE_range_of_a_for_line_separating_points_l2838_283834

/-- Given points A and B on opposite sides of the line 3x + 2y + a = 0, 
    prove that the range of a is (-19, -9) -/
theorem range_of_a_for_line_separating_points 
  (A B : ℝ × ℝ) 
  (h_A : A = (1, 3)) 
  (h_B : B = (5, 2)) 
  (h_opposite : (3 * A.1 + 2 * A.2 + a) * (3 * B.1 + 2 * B.2 + a) < 0) :
  ∀ a : ℝ, (a > -19 ∧ a < -9) ↔ 
    ∃ (x y : ℝ), (3 * x + 2 * y + a = 0 ∧ 
      (3 * A.1 + 2 * A.2 + a) * (3 * B.1 + 2 * B.2 + a) < 0) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_for_line_separating_points_l2838_283834


namespace NUMINAMATH_CALUDE_min_product_positive_numbers_l2838_283896

theorem min_product_positive_numbers (x₁ x₂ x₃ : ℝ) :
  x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 →
  x₁ + x₂ + x₃ = 4 →
  (∀ (i j : Fin 3), i ≠ j → 2 * x₁^2 + 2 * x₂^2 - 5 * x₁ * x₂ ≤ 0) →
  (∀ (i j : Fin 3), i ≠ j → 2 * x₁^2 + 2 * x₃^2 - 5 * x₁ * x₃ ≤ 0) →
  (∀ (i j : Fin 3), i ≠ j → 2 * x₂^2 + 2 * x₃^2 - 5 * x₂ * x₃ ≤ 0) →
  x₁ * x₂ * x₃ ≥ 2 ∧ ∃ (y₁ y₂ y₃ : ℝ), y₁ > 0 ∧ y₂ > 0 ∧ y₃ > 0 ∧
    y₁ + y₂ + y₃ = 4 ∧
    (∀ (i j : Fin 3), i ≠ j → 2 * y₁^2 + 2 * y₂^2 - 5 * y₁ * y₂ ≤ 0) ∧
    (∀ (i j : Fin 3), i ≠ j → 2 * y₁^2 + 2 * y₃^2 - 5 * y₁ * y₃ ≤ 0) ∧
    (∀ (i j : Fin 3), i ≠ j → 2 * y₂^2 + 2 * y₃^2 - 5 * y₂ * y₃ ≤ 0) ∧
    y₁ * y₂ * y₃ = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_product_positive_numbers_l2838_283896


namespace NUMINAMATH_CALUDE_bacteria_count_after_six_hours_l2838_283842

/-- The number of bacteria at time n -/
def bacteria : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | (n + 2) => bacteria (n + 1) + bacteria n

/-- The time in half-hour units after which we want to count bacteria -/
def target_time : ℕ := 12

theorem bacteria_count_after_six_hours :
  bacteria target_time = 233 := by
  sorry

end NUMINAMATH_CALUDE_bacteria_count_after_six_hours_l2838_283842


namespace NUMINAMATH_CALUDE_bill_denomination_l2838_283831

theorem bill_denomination (total_amount : ℕ) (num_bills : ℕ) (h1 : total_amount = 45) (h2 : num_bills = 9) :
  total_amount / num_bills = 5 := by
sorry

end NUMINAMATH_CALUDE_bill_denomination_l2838_283831


namespace NUMINAMATH_CALUDE_flag_covering_l2838_283865

theorem flag_covering (grid_height : Nat) (grid_width : Nat) (flag_count : Nat) :
  grid_height = 9 →
  grid_width = 18 →
  flag_count = 18 →
  (∃ (ways_to_place_flag : Nat), ways_to_place_flag = 2) →
  (∃ (total_ways : Nat), total_ways = 2^flag_count) :=
by sorry

end NUMINAMATH_CALUDE_flag_covering_l2838_283865


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_35_l2838_283883

theorem smallest_four_digit_divisible_by_35 :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 → n ≥ 1050 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_35_l2838_283883


namespace NUMINAMATH_CALUDE_factor_count_of_M_l2838_283802

/-- The number of natural-number factors of M, where M = 2^4 * 3^3 * 5^2 * 7^1 -/
def number_of_factors (M : ℕ) : ℕ :=
  5 * 4 * 3 * 2

theorem factor_count_of_M :
  let M : ℕ := 2^4 * 3^3 * 5^2 * 7^1
  number_of_factors M = 120 := by
  sorry

end NUMINAMATH_CALUDE_factor_count_of_M_l2838_283802


namespace NUMINAMATH_CALUDE_total_children_l2838_283891

theorem total_children (happy : ℕ) (sad : ℕ) (neutral : ℕ) 
  (boys : ℕ) (girls : ℕ) (happy_boys : ℕ) (sad_girls : ℕ) (neutral_boys : ℕ) :
  happy = 30 →
  sad = 10 →
  neutral = 20 →
  boys = 19 →
  girls = 41 →
  happy_boys = 6 →
  sad_girls = 4 →
  neutral_boys = 7 →
  boys + girls = 60 :=
by sorry

end NUMINAMATH_CALUDE_total_children_l2838_283891


namespace NUMINAMATH_CALUDE_fraction_equality_l2838_283866

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (5*x + 2*y) / (x - 5*y) = -3) : 
  (x + 5*y) / (5*x - y) = 53/57 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l2838_283866


namespace NUMINAMATH_CALUDE_increasing_function_iff_a_in_range_l2838_283804

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 - a) * x - a else Real.log x / Real.log a

-- State the theorem
theorem increasing_function_iff_a_in_range :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) ↔ (3/2 ≤ a ∧ a < 3) :=
by sorry

end NUMINAMATH_CALUDE_increasing_function_iff_a_in_range_l2838_283804


namespace NUMINAMATH_CALUDE_x_value_proof_l2838_283871

theorem x_value_proof (x : ℚ) 
  (eq1 : 8 * x^2 + 7 * x - 1 = 0)
  (eq2 : 24 * x^2 + 53 * x - 7 = 0) :
  x = 1/8 := by
sorry

end NUMINAMATH_CALUDE_x_value_proof_l2838_283871


namespace NUMINAMATH_CALUDE_parabola_coefficient_l2838_283849

/-- Proves that for a parabola y = x^2 + bx + c passing through (1, -1) and (3, 9), c = -3 -/
theorem parabola_coefficient (b c : ℝ) : 
  (1^2 + b*1 + c = -1) → 
  (3^2 + b*3 + c = 9) → 
  c = -3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_coefficient_l2838_283849


namespace NUMINAMATH_CALUDE_fruit_salad_weight_l2838_283857

/-- The total weight of fruit in Scarlett's fruit salad is 1.85 pounds. -/
theorem fruit_salad_weight :
  let melon : ℚ := 35/100
  let berries : ℚ := 48/100
  let grapes : ℚ := 29/100
  let pineapple : ℚ := 56/100
  let oranges : ℚ := 17/100
  melon + berries + grapes + pineapple + oranges = 185/100 := by
  sorry

end NUMINAMATH_CALUDE_fruit_salad_weight_l2838_283857


namespace NUMINAMATH_CALUDE_jills_salary_l2838_283816

/-- Represents a person's monthly financial allocation --/
structure MonthlyFinances where
  netSalary : ℝ
  discretionaryIncome : ℝ
  vacationFund : ℝ
  savings : ℝ
  socializing : ℝ
  charitable : ℝ

/-- Conditions for Jill's financial allocation --/
def JillsFinances (m : MonthlyFinances) : Prop :=
  m.discretionaryIncome = m.netSalary / 5 ∧
  m.vacationFund = 0.3 * m.discretionaryIncome ∧
  m.savings = 0.2 * m.discretionaryIncome ∧
  m.socializing = 0.35 * m.discretionaryIncome ∧
  m.charitable = 99

/-- Theorem stating that under the given conditions, Jill's net monthly salary is $3300 --/
theorem jills_salary (m : MonthlyFinances) (h : JillsFinances m) : m.netSalary = 3300 := by
  sorry

end NUMINAMATH_CALUDE_jills_salary_l2838_283816


namespace NUMINAMATH_CALUDE_triangle_area_l2838_283840

/-- The area of a right triangle with base 2 and height (12 - p) is equal to 12 - p. -/
theorem triangle_area (p : ℝ) : 
  (1 / 2 : ℝ) * 2 * (12 - p) = 12 - p :=
sorry

end NUMINAMATH_CALUDE_triangle_area_l2838_283840


namespace NUMINAMATH_CALUDE_binomial_20_5_l2838_283820

theorem binomial_20_5 : Nat.choose 20 5 = 11628 := by sorry

end NUMINAMATH_CALUDE_binomial_20_5_l2838_283820


namespace NUMINAMATH_CALUDE_cuboid_color_is_blue_l2838_283875

/-- Represents a cube with colored faces -/
structure ColoredCube where
  red_faces : Fin 6
  blue_faces : Fin 6
  yellow_faces : Fin 6
  face_sum : red_faces + blue_faces + yellow_faces = 6

/-- Represents the arrangement of cubes in a photo -/
structure CubeArrangement where
  red_visible : Nat
  blue_visible : Nat
  yellow_visible : Nat
  total_visible : red_visible + blue_visible + yellow_visible = 8

/-- The set of four cubes -/
def cube_set : Finset ColoredCube := sorry

/-- The three different arrangements in the colored photos -/
def arrangements : Finset CubeArrangement := sorry

theorem cuboid_color_is_blue 
  (h1 : ∀ c ∈ cube_set, c.red_faces + c.blue_faces + c.yellow_faces = 6)
  (h2 : cube_set.card = 4)
  (h3 : ∀ a ∈ arrangements, a.red_visible + a.blue_visible + a.yellow_visible = 8)
  (h4 : arrangements.card = 3)
  (h5 : (arrangements.sum (λ a => a.red_visible)) = 8)
  (h6 : (arrangements.sum (λ a => a.blue_visible)) = 8)
  (h7 : (arrangements.sum (λ a => a.yellow_visible)) = 8)
  (h8 : ∃ a ∈ arrangements, a.red_visible = 2)
  (h9 : ∃ c ∈ cube_set, c.yellow_faces = 0) :
  ∀ c ∈ cube_set, c.blue_faces ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_cuboid_color_is_blue_l2838_283875


namespace NUMINAMATH_CALUDE_alex_and_sam_speeds_l2838_283812

-- Define the variables
def alex_downstream_distance : ℝ := 36
def alex_downstream_time : ℝ := 6
def alex_upstream_time : ℝ := 9
def sam_downstream_distance : ℝ := 48
def sam_downstream_time : ℝ := 8
def sam_upstream_time : ℝ := 12

-- Define the theorem
theorem alex_and_sam_speeds :
  ∃ (alex_speed sam_speed current_speed : ℝ),
    alex_speed > 0 ∧ sam_speed > 0 ∧
    (alex_speed + current_speed) * alex_downstream_time = alex_downstream_distance ∧
    (alex_speed - current_speed) * alex_upstream_time = alex_downstream_distance ∧
    (sam_speed + current_speed) * sam_downstream_time = sam_downstream_distance ∧
    (sam_speed - current_speed) * sam_upstream_time = sam_downstream_distance ∧
    alex_speed = 5 ∧ sam_speed = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_alex_and_sam_speeds_l2838_283812


namespace NUMINAMATH_CALUDE_inequality_proof_l2838_283837

theorem inequality_proof (a b : ℝ) (ha : a > 1) (hb : -1 < b ∧ b < 0) : a * b^2 > b^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2838_283837


namespace NUMINAMATH_CALUDE_leahs_coins_value_l2838_283881

theorem leahs_coins_value :
  ∀ (p d : ℕ),
  p + d = 15 →
  p = d + 1 →
  p * 1 + d * 10 = 87 :=
by sorry

end NUMINAMATH_CALUDE_leahs_coins_value_l2838_283881


namespace NUMINAMATH_CALUDE_circle_line_properties_l2838_283858

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x + 1)^2 + (y - 2)^2 = 25

-- Define the line l
def line_l (m x y : ℝ) : Prop := (3*m + 1)*x + (m + 1)*y - 5*m - 3 = 0

-- Theorem statement
theorem circle_line_properties :
  -- Line l intersects circle C
  (∃ (m x y : ℝ), circle_C x y ∧ line_l m x y) ∧
  -- The chord length intercepted by circle C on the y-axis is 4√6
  (∃ (y1 y2 : ℝ), circle_C 0 y1 ∧ circle_C 0 y2 ∧ y2 - y1 = 4 * Real.sqrt 6) ∧
  -- When the chord length intercepted by circle C is the shortest, the equation of line l is x=1
  (∃ (m : ℝ), ∀ (x y : ℝ), line_l m x y → x = 1) :=
sorry

end NUMINAMATH_CALUDE_circle_line_properties_l2838_283858


namespace NUMINAMATH_CALUDE_complex_sum_equality_l2838_283889

theorem complex_sum_equality : 
  12 * Complex.exp (3 * Real.pi * Complex.I / 13) + 
  12 * Complex.exp (7 * Real.pi * Complex.I / 26) = 
  24 * Real.cos (Real.pi / 26) * Complex.exp (19 * Real.pi * Complex.I / 52) := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_equality_l2838_283889


namespace NUMINAMATH_CALUDE_middle_part_of_proportional_distribution_l2838_283833

theorem middle_part_of_proportional_distribution (total : ℚ) (r1 r2 r3 : ℚ) :
  total = 120 →
  r1 = 1 →
  r2 = 1/4 →
  r3 = 1/8 →
  (r2 * total) / (r1 + r2 + r3) = 240/11 := by
  sorry

end NUMINAMATH_CALUDE_middle_part_of_proportional_distribution_l2838_283833


namespace NUMINAMATH_CALUDE_edric_working_days_l2838_283878

/-- Calculates the number of working days per week given monthly salary, hours per day, and hourly rate -/
def working_days_per_week (monthly_salary : ℕ) (hours_per_day : ℕ) (hourly_rate : ℕ) : ℚ :=
  (monthly_salary : ℚ) / 4 / (hours_per_day * hourly_rate)

/-- Theorem: Given Edric's monthly salary, hours per day, and hourly rate, he works 6 days a week -/
theorem edric_working_days : working_days_per_week 576 8 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_edric_working_days_l2838_283878


namespace NUMINAMATH_CALUDE_largest_b_for_divisibility_l2838_283886

def is_divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

def five_digit_number (b : ℕ) : ℕ := 48000 + b * 100 + 56

theorem largest_b_for_divisibility :
  ∀ b : ℕ, b ≤ 9 →
    (is_divisible_by_4 (five_digit_number b) → b ≤ 8) ∧
    is_divisible_by_4 (five_digit_number 8) :=
by sorry

end NUMINAMATH_CALUDE_largest_b_for_divisibility_l2838_283886


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2838_283853

theorem arithmetic_calculation : 
  (1.5 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) : ℝ) = 1200 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2838_283853


namespace NUMINAMATH_CALUDE_multiplicative_inverse_modulo_million_l2838_283806

theorem multiplicative_inverse_modulo_million : 
  let A : ℕ := 123456
  let B : ℕ := 153846
  let N : ℕ := 500000
  let M : ℕ := 1000000
  (A * B * N) % M = 1 := by
  sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_modulo_million_l2838_283806


namespace NUMINAMATH_CALUDE_oddSum_not_prime_l2838_283897

def oddSum (n : Nat) : Nat :=
  List.sum (List.map (fun i => 2 * i - 1) (List.range n))

theorem oddSum_not_prime (n : Nat) (h : 2 ≤ n ∧ n ≤ 5) : ¬ Nat.Prime (oddSum n) := by
  sorry

end NUMINAMATH_CALUDE_oddSum_not_prime_l2838_283897


namespace NUMINAMATH_CALUDE_inequality_proof_l2838_283882

theorem inequality_proof (x : ℝ) : (2 * x - 1) / 3 ≥ 1 → x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2838_283882


namespace NUMINAMATH_CALUDE_proper_subsets_of_A_l2838_283848

def U : Finset ℕ := {0,1,2,3,4,5}

def C_U_A : Finset ℕ := {1,2,3}

def A : Finset ℕ := U \ C_U_A

theorem proper_subsets_of_A : Finset.card (Finset.powerset A \ {A}) = 7 := by
  sorry

end NUMINAMATH_CALUDE_proper_subsets_of_A_l2838_283848


namespace NUMINAMATH_CALUDE_roxy_garden_problem_l2838_283845

def garden_problem (initial_flowering : ℕ) (initial_fruiting_factor : ℕ) 
  (bought_flowering : ℕ) (bought_fruiting : ℕ) 
  (given_fruiting : ℕ) (total_remaining : ℕ) : Prop :=
  let initial_fruiting := initial_flowering * initial_fruiting_factor
  let after_purchase_flowering := initial_flowering + bought_flowering
  let after_purchase_fruiting := initial_fruiting + bought_fruiting
  let remaining_fruiting := after_purchase_fruiting - given_fruiting
  let remaining_flowering := total_remaining - remaining_fruiting
  let given_flowering := after_purchase_flowering - remaining_flowering
  given_flowering = 1

theorem roxy_garden_problem : 
  garden_problem 7 2 3 2 4 21 :=
sorry

end NUMINAMATH_CALUDE_roxy_garden_problem_l2838_283845


namespace NUMINAMATH_CALUDE_abc_value_l2838_283803

theorem abc_value (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a * b = 45 * Real.rpow 3 (1/3))
  (hac : a * c = 63 * Real.rpow 3 (1/3))
  (hbc : b * c = 28 * Real.rpow 3 (1/3)) :
  a * b * c = 630 := by
sorry

end NUMINAMATH_CALUDE_abc_value_l2838_283803


namespace NUMINAMATH_CALUDE_angle_between_vectors_l2838_283838

theorem angle_between_vectors (a b : ℝ × ℝ) 
  (h1 : a.1 * b.1 + a.2 * b.2 = -1)  -- dot product condition
  (h2 : Real.sqrt (a.1^2 + a.2^2) = 2)  -- magnitude of a
  (h3 : Real.sqrt (b.1^2 + b.2^2) = 1)  -- magnitude of b
  : Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))) = 2*π/3 := by
  sorry

end NUMINAMATH_CALUDE_angle_between_vectors_l2838_283838


namespace NUMINAMATH_CALUDE_exactly_two_b_values_l2838_283811

-- Define the quadratic function
def f (b : ℤ) (x : ℤ) : ℤ := x^2 + b*x + 3

-- Define a predicate for when f(b,x) ≤ 0
def satisfies_inequality (b : ℤ) (x : ℤ) : Prop := f b x ≤ 0

-- Define a predicate for when b gives exactly three integer solutions
def has_three_solutions (b : ℤ) : Prop :=
  ∃ x₁ x₂ x₃ : ℤ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    satisfies_inequality b x₁ ∧
    satisfies_inequality b x₂ ∧
    satisfies_inequality b x₃ ∧
    ∀ x : ℤ, satisfies_inequality b x → (x = x₁ ∨ x = x₂ ∨ x = x₃)

-- The main theorem
theorem exactly_two_b_values :
  ∃! s : Finset ℤ, s.card = 2 ∧ ∀ b : ℤ, b ∈ s ↔ has_three_solutions b :=
sorry

end NUMINAMATH_CALUDE_exactly_two_b_values_l2838_283811


namespace NUMINAMATH_CALUDE_perpendicular_vectors_t_value_l2838_283830

/-- Two-dimensional vector -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Dot product of two 2D vectors -/
def dotProduct (v w : Vector2D) : ℝ := v.x * w.x + v.y * w.y

/-- Perpendicularity of two 2D vectors -/
def isPerpendicular (v w : Vector2D) : Prop := dotProduct v w = 0

theorem perpendicular_vectors_t_value :
  ∀ t : ℝ,
  let a : Vector2D := ⟨t, 1⟩
  let b : Vector2D := ⟨1, 2⟩
  isPerpendicular a b → t = -2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_t_value_l2838_283830


namespace NUMINAMATH_CALUDE_circle_through_points_on_line_equation_l2838_283856

/-- A circle passing through two points with its center on a given line -/
def CircleThroughPointsOnLine (A B C : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let (a, b) := C
  (x₁ - a)^2 + (y₁ - b)^2 = (x₂ - a)^2 + (y₂ - b)^2 ∧ a + b = 2

/-- The standard equation of a circle -/
def StandardCircleEquation (C : ℝ × ℝ) (r : ℝ) (x y : ℝ) : Prop :=
  let (a, b) := C
  (x - a)^2 + (y - b)^2 = r^2

theorem circle_through_points_on_line_equation :
  ∀ (C : ℝ × ℝ),
  CircleThroughPointsOnLine (1, -1) (-1, 1) C →
  ∃ (x y : ℝ), StandardCircleEquation C 2 x y :=
by sorry

end NUMINAMATH_CALUDE_circle_through_points_on_line_equation_l2838_283856


namespace NUMINAMATH_CALUDE_fibSum_eq_five_nineteenths_l2838_283819

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- The sum of the Fibonacci series divided by powers of 5 -/
noncomputable def fibSum : ℝ := ∑' n, (fib n : ℝ) / 5^n

/-- Theorem stating that the sum of the Fibonacci series divided by powers of 5 equals 5/19 -/
theorem fibSum_eq_five_nineteenths : fibSum = 5 / 19 := by sorry

end NUMINAMATH_CALUDE_fibSum_eq_five_nineteenths_l2838_283819


namespace NUMINAMATH_CALUDE_circular_table_seating_l2838_283899

-- Define the number of people and seats
def total_people : ℕ := 9
def table_seats : ℕ := 7

-- Define the function to calculate the number of seating arrangements
def seating_arrangements (n : ℕ) (k : ℕ) : ℕ :=
  (Nat.choose n (n - k)) * (Nat.factorial (k - 1))

-- Theorem statement
theorem circular_table_seating :
  seating_arrangements total_people table_seats = 25920 :=
sorry

end NUMINAMATH_CALUDE_circular_table_seating_l2838_283899


namespace NUMINAMATH_CALUDE_parabola_symmetry_axis_part1_parabola_symmetry_axis_part2_l2838_283885

-- Define the parabola and its properties
def Parabola (a b c : ℝ) (h : a > 0) :=
  {f : ℝ → ℝ | ∀ x, f x = a * x^2 + b * x + c}

def AxisOfSymmetry (t : ℝ) (p : Parabola a b c h) :=
  t = -b / (2 * a)

-- Theorem for part (1)
theorem parabola_symmetry_axis_part1
  (a b c : ℝ) (h : a > 0) (p : Parabola a b c h) (t : ℝ) :
  AxisOfSymmetry t p →
  (a * 1^2 + b * 1 + c = a * 2^2 + b * 2 + c) →
  t = 3/2 := by sorry

-- Theorem for part (2)
theorem parabola_symmetry_axis_part2
  (a b c : ℝ) (h : a > 0) (p : Parabola a b c h) (t : ℝ) :
  AxisOfSymmetry t p →
  (∀ x₁ x₂, 0 < x₁ → x₁ < 1 → 1 < x₂ → x₂ < 2 →
    a * x₁^2 + b * x₁ + c < a * x₂^2 + b * x₂ + c) →
  t ≤ 1/2 := by sorry

end NUMINAMATH_CALUDE_parabola_symmetry_axis_part1_parabola_symmetry_axis_part2_l2838_283885


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l2838_283876

theorem intersection_implies_a_value (a : ℝ) : 
  let A : Set ℝ := {a^2, a+1, -3}
  let B : Set ℝ := {a-3, 3*a-1, a^2+1}
  A ∩ B = {-3} → a = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l2838_283876


namespace NUMINAMATH_CALUDE_leftover_coins_value_l2838_283835

def quarters_per_roll : ℕ := 50
def dimes_per_roll : ℕ := 40
def michael_quarters : ℕ := 95
def michael_dimes : ℕ := 172
def anna_quarters : ℕ := 140
def anna_dimes : ℕ := 287
def quarter_value : ℚ := 0.25
def dime_value : ℚ := 0.10

theorem leftover_coins_value :
  let total_quarters := michael_quarters + anna_quarters
  let total_dimes := michael_dimes + anna_dimes
  let leftover_quarters := total_quarters % quarters_per_roll
  let leftover_dimes := total_dimes % dimes_per_roll
  let leftover_value := (leftover_quarters : ℚ) * quarter_value + (leftover_dimes : ℚ) * dime_value
  leftover_value = 10.65 := by sorry

end NUMINAMATH_CALUDE_leftover_coins_value_l2838_283835


namespace NUMINAMATH_CALUDE_zeros_of_f_l2838_283855

-- Define the function f(x) = x^3 - 3x + 2
def f (x : ℝ) : ℝ := x^3 - 3*x + 2

-- Theorem stating that 1 and -2 are the zeros of f
theorem zeros_of_f : 
  (∃ x : ℝ, f x = 0) ∧ (∀ x : ℝ, f x = 0 → x = 1 ∨ x = -2) ∧ f 1 = 0 ∧ f (-2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_zeros_of_f_l2838_283855


namespace NUMINAMATH_CALUDE_school_play_seating_l2838_283894

/-- Given a school play seating arrangement, prove the number of unoccupied seats. -/
theorem school_play_seating (rows : ℕ) (chairs_per_row : ℕ) (occupied_seats : ℕ) 
  (h1 : rows = 40)
  (h2 : chairs_per_row = 20)
  (h3 : occupied_seats = 790) :
  rows * chairs_per_row - occupied_seats = 10 := by
  sorry

end NUMINAMATH_CALUDE_school_play_seating_l2838_283894


namespace NUMINAMATH_CALUDE_product_in_N_l2838_283836

def M : Set ℤ := {x | ∃ m : ℤ, x = 3 * m + 1}
def N : Set ℤ := {y | ∃ n : ℤ, y = 3 * n + 2}

theorem product_in_N (x y : ℤ) (hx : x ∈ M) (hy : y ∈ N) : x * y ∈ N := by
  sorry

end NUMINAMATH_CALUDE_product_in_N_l2838_283836


namespace NUMINAMATH_CALUDE_octagon_diagonal_intersection_probability_l2838_283864

/-- A regular octagon is an 8-sided polygon with all sides equal and all angles equal. -/
def RegularOctagon : Type := Unit

/-- The number of diagonals in a regular octagon. -/
def num_diagonals (octagon : RegularOctagon) : ℕ := 20

/-- The number of pairs of diagonals in a regular octagon. -/
def num_diagonal_pairs (octagon : RegularOctagon) : ℕ := 190

/-- The number of pairs of intersecting diagonals in a regular octagon. -/
def num_intersecting_pairs (octagon : RegularOctagon) : ℕ := 70

/-- The probability that two randomly chosen diagonals in a regular octagon intersect inside the octagon. -/
theorem octagon_diagonal_intersection_probability (octagon : RegularOctagon) :
  (num_intersecting_pairs octagon : ℚ) / (num_diagonal_pairs octagon) = 7 / 19 := by
  sorry

end NUMINAMATH_CALUDE_octagon_diagonal_intersection_probability_l2838_283864


namespace NUMINAMATH_CALUDE_range_of_m_for_necessary_condition_l2838_283825

theorem range_of_m_for_necessary_condition : 
  ∀ m : ℝ, 
  (∀ x : ℝ, x^2 - 2*x - 3 > 0 → (x < m - 1 ∨ x > m + 1)) ∧ 
  (∃ x : ℝ, (x < m - 1 ∨ x > m + 1) ∧ x^2 - 2*x - 3 ≤ 0) ↔ 
  m ∈ Set.Icc 0 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_for_necessary_condition_l2838_283825


namespace NUMINAMATH_CALUDE_rosie_pies_l2838_283800

/-- The number of apples required to make one pie -/
def apples_per_pie : ℕ := 5

/-- The total number of apples Rosie has -/
def total_apples : ℕ := 32

/-- The maximum number of whole pies that can be made -/
def max_pies : ℕ := total_apples / apples_per_pie

theorem rosie_pies :
  max_pies = 6 :=
sorry

end NUMINAMATH_CALUDE_rosie_pies_l2838_283800


namespace NUMINAMATH_CALUDE_average_speed_two_segment_trip_l2838_283888

/-- Given a trip with two segments, prove that the average speed is as calculated -/
theorem average_speed_two_segment_trip (distance1 : ℝ) (speed1 : ℝ) (distance2 : ℝ) (speed2 : ℝ) 
  (h1 : distance1 = 360)
  (h2 : speed1 = 60)
  (h3 : distance2 = 120)
  (h4 : speed2 = 40) :
  (distance1 + distance2) / ((distance1 / speed1) + (distance2 / speed2)) = 480 / 9 := by
sorry

end NUMINAMATH_CALUDE_average_speed_two_segment_trip_l2838_283888


namespace NUMINAMATH_CALUDE_sequence_general_term_l2838_283872

theorem sequence_general_term (a : ℕ → ℝ) (h1 : a 1 = 1) 
    (h2 : ∀ n : ℕ, n ≥ 2 → a n - a (n-1) = 2) : 
    ∀ n : ℕ, a n = 2 * n - 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_general_term_l2838_283872


namespace NUMINAMATH_CALUDE_pams_bags_to_geralds_bags_l2838_283879

/-- Represents the number of apples in each of Gerald's bags -/
def geralds_bag_size : ℕ := 40

/-- Represents the total number of apples Pam has -/
def pams_total_apples : ℕ := 1200

/-- Represents the number of bags Pam has -/
def pams_bag_count : ℕ := 10

/-- Theorem stating that each of Pam's bags equates to 3 of Gerald's bags -/
theorem pams_bags_to_geralds_bags : 
  (pams_total_apples / pams_bag_count) / geralds_bag_size = 3 := by
  sorry

end NUMINAMATH_CALUDE_pams_bags_to_geralds_bags_l2838_283879


namespace NUMINAMATH_CALUDE_problem_statement_l2838_283844

theorem problem_statement (a b c d : ℝ) 
  (h : (a - b) * (c - d) / ((b - c) * (d - a)) = 2 / 5) :
  (a - c) * (b - d) / ((a - b) * (c - d)) = -3 / 2 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2838_283844


namespace NUMINAMATH_CALUDE_cubic_roots_cube_l2838_283805

theorem cubic_roots_cube (u v w : ℂ) :
  (u^3 + v^3 + w^3 = 54) →
  (u^3 * v^3 + v^3 * w^3 + w^3 * u^3 = -89) →
  (u^3 * v^3 * w^3 = 27) →
  (u + v + w = 5) →
  (u * v + v * w + w * u = 4) →
  (u * v * w = 3) →
  (u^3 - 5 * u^2 + 4 * u - 3 = 0) →
  (v^3 - 5 * v^2 + 4 * v - 3 = 0) →
  (w^3 - 5 * w^2 + 4 * w - 3 = 0) →
  ∀ (x : ℂ), x^3 - 54 * x^2 - 89 * x - 27 = 0 ↔ (x = u^3 ∨ x = v^3 ∨ x = w^3) := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_cube_l2838_283805


namespace NUMINAMATH_CALUDE_fraction_problem_l2838_283810

theorem fraction_problem :
  let f₁ : ℚ := 75 / 34
  let f₂ : ℚ := 70 / 51
  (f₁ - f₂ = 5 / 6) ∧
  (Nat.gcd 75 70 = 75 - 70) ∧
  (Nat.lcm 75 70 = 1050) ∧
  (∀ a b c d : ℕ, (a / b : ℚ) = f₁ ∧ (c / d : ℚ) = f₂ → Nat.gcd a c = 1 ∧ Nat.gcd b d = 1) :=
by sorry

end NUMINAMATH_CALUDE_fraction_problem_l2838_283810


namespace NUMINAMATH_CALUDE_third_year_sample_size_l2838_283808

/-- Calculates the number of students to be sampled from a specific grade in stratified sampling -/
def stratified_sample_size (total_population : ℕ) (grade_population : ℕ) (total_sample_size : ℕ) : ℕ :=
  (grade_population * total_sample_size) / total_population

/-- Proves that the number of third-year students to be sampled is 21 -/
theorem third_year_sample_size :
  let total_population : ℕ := 3000
  let third_year_population : ℕ := 1050
  let total_sample_size : ℕ := 60
  stratified_sample_size total_population third_year_population total_sample_size = 21 := by
  sorry


end NUMINAMATH_CALUDE_third_year_sample_size_l2838_283808
