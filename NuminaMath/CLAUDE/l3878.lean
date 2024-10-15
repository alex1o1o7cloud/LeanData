import Mathlib

namespace NUMINAMATH_CALUDE_polynomial_value_at_negative_one_l3878_387804

theorem polynomial_value_at_negative_one (r : ℝ) : 
  (fun x : ℝ => 3 * x^4 - 2 * x^3 + x^2 + 4 * x + r) (-1) = 0 → r = -2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_at_negative_one_l3878_387804


namespace NUMINAMATH_CALUDE_fourth_power_sum_l3878_387852

theorem fourth_power_sum (a b c : ℝ) 
  (sum_1 : a + b + c = 1)
  (sum_2 : a^2 + b^2 + c^2 = 2)
  (sum_3 : a^3 + b^3 + c^3 = 3) :
  a^4 + b^4 + c^4 = 25/6 := by sorry

end NUMINAMATH_CALUDE_fourth_power_sum_l3878_387852


namespace NUMINAMATH_CALUDE_simplify_expression_l3878_387872

theorem simplify_expression (x : ℝ) : (2 * x + 20) + (150 * x + 20) = 152 * x + 40 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3878_387872


namespace NUMINAMATH_CALUDE_bill_selling_price_l3878_387815

theorem bill_selling_price (purchase_price : ℝ) : 
  (purchase_price * 1.1 : ℝ) = 550 ∧ 
  (purchase_price * 0.9 * 1.3 : ℝ) - (purchase_price * 1.1 : ℝ) = 35 :=
by sorry

end NUMINAMATH_CALUDE_bill_selling_price_l3878_387815


namespace NUMINAMATH_CALUDE_one_match_probability_l3878_387842

/-- The number of balls and boxes -/
def n : ℕ := 4

/-- The total number of ways to distribute balls into boxes -/
def total_arrangements : ℕ := n.factorial

/-- The number of ways to distribute balls with exactly one color match -/
def matching_arrangements : ℕ := n * ((n - 1).factorial)

/-- The probability of exactly one ball matching its box color -/
def probability_one_match : ℚ := matching_arrangements / total_arrangements

theorem one_match_probability :
  probability_one_match = 1/3 :=
sorry

end NUMINAMATH_CALUDE_one_match_probability_l3878_387842


namespace NUMINAMATH_CALUDE_number_calculation_l3878_387808

theorem number_calculation (n x : ℝ) (h1 : x = 0.8999999999999999) (h2 : n / x = 0.01) :
  n = 0.008999999999999999 := by
sorry

end NUMINAMATH_CALUDE_number_calculation_l3878_387808


namespace NUMINAMATH_CALUDE_marble_probability_l3878_387839

theorem marble_probability (total : ℕ) (p_white p_green : ℚ) 
  (h_total : total = 84)
  (h_white : p_white = 1/4)
  (h_green : p_green = 2/7) :
  1 - p_white - p_green = 13/28 := by
  sorry

end NUMINAMATH_CALUDE_marble_probability_l3878_387839


namespace NUMINAMATH_CALUDE_second_shipment_weight_l3878_387819

/-- Represents the weight of couscous shipments and dishes made at a Moroccan restaurant -/
structure CouscousShipments where
  first_shipment : ℕ
  second_shipment : ℕ
  third_shipment : ℕ
  num_dishes : ℕ
  couscous_per_dish : ℕ

/-- Theorem stating the weight of the second shipment of couscous -/
theorem second_shipment_weight (c : CouscousShipments) 
  (h1 : c.first_shipment = 7)
  (h2 : c.third_shipment = 45)
  (h3 : c.num_dishes = 13)
  (h4 : c.couscous_per_dish = 5)
  (h5 : c.first_shipment + c.second_shipment + c.third_shipment = c.num_dishes * c.couscous_per_dish) :
  c.second_shipment = 13 := by
  sorry

end NUMINAMATH_CALUDE_second_shipment_weight_l3878_387819


namespace NUMINAMATH_CALUDE_rounding_accuracy_of_1_35_billion_l3878_387830

theorem rounding_accuracy_of_1_35_billion :
  ∃ n : ℕ, (1350000000 : ℕ) = n * 10000000 ∧ n % 10 ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_rounding_accuracy_of_1_35_billion_l3878_387830


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l3878_387869

-- Define the universal set U
def U : Set ℕ := {x : ℕ | x ≥ 2}

-- Define set A
def A : Set ℕ := {x : ℕ | x^2 ≥ 5}

-- Theorem statement
theorem complement_of_A_in_U : (U \ A) = {2} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l3878_387869


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_l3878_387827

open Set

def M : Set ℝ := {x | x > 2}
def P : Set ℝ := {x | x < 3}

theorem necessary_not_sufficient :
  (∀ x, x ∈ M ∩ P → (x ∈ M ∨ x ∈ P)) ∧
  (∃ x, (x ∈ M ∨ x ∈ P) ∧ x ∉ M ∩ P) :=
by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_l3878_387827


namespace NUMINAMATH_CALUDE_cubic_root_sum_cubes_l3878_387870

theorem cubic_root_sum_cubes (a b c : ℂ) : 
  (a^3 - 2*a^2 + 3*a - 4 = 0) → 
  (b^3 - 2*b^2 + 3*b - 4 = 0) → 
  (c^3 - 2*c^2 + 3*c - 4 = 0) → 
  a^3 + b^3 + c^3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_cubes_l3878_387870


namespace NUMINAMATH_CALUDE_john_twice_sam_age_l3878_387834

def john_age (sam_age : ℕ) : ℕ := 3 * sam_age

def sam_current_age : ℕ := 7 + 2

theorem john_twice_sam_age (years : ℕ) : 
  john_age sam_current_age + years = 2 * (sam_current_age + years) → years = 9 := by
  sorry

end NUMINAMATH_CALUDE_john_twice_sam_age_l3878_387834


namespace NUMINAMATH_CALUDE_complex_difference_of_eighth_powers_l3878_387871

theorem complex_difference_of_eighth_powers : (2 + Complex.I) ^ 8 - (2 - Complex.I) ^ 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_difference_of_eighth_powers_l3878_387871


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_two_l3878_387851

theorem reciprocal_of_negative_two :
  ∀ x : ℚ, x * (-2) = 1 → x = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_two_l3878_387851


namespace NUMINAMATH_CALUDE_prize_winning_condition_xiao_feng_inequality_l3878_387857

/-- Represents the school intelligence competition --/
structure Competition where
  total_questions : ℕ
  correct_points : ℤ
  incorrect_points : ℤ
  prize_threshold : ℤ

/-- Represents a student participating in the competition --/
structure Student where
  correct_answers : ℕ
  won_prize : Prop

/-- The specific competition described in the problem --/
def school_competition : Competition :=
  { total_questions := 20
  , correct_points := 5
  , incorrect_points := -2
  , prize_threshold := 75 }

/-- Theorem stating the condition for winning a prize --/
theorem prize_winning_condition (s : Student) (c : Competition) 
  (h1 : s.won_prize) 
  (h2 : s.correct_answers ≤ c.total_questions) :
  c.correct_points * s.correct_answers + 
  c.incorrect_points * (c.total_questions - s.correct_answers) > 
  c.prize_threshold := by
  sorry

/-- Theorem for Xiao Feng's specific case --/
theorem xiao_feng_inequality (x : ℕ) :
  x ≤ school_competition.total_questions →
  5 * x - 2 * (20 - x) > 75 := by
  sorry

end NUMINAMATH_CALUDE_prize_winning_condition_xiao_feng_inequality_l3878_387857


namespace NUMINAMATH_CALUDE_cone_volume_from_lateral_surface_l3878_387812

/-- Given a cone whose lateral surface, when unfolded, forms a semicircle with an area of 2π,
    the volume of the cone is (√3/3)π. -/
theorem cone_volume_from_lateral_surface (l r h : ℝ) : 
  l > 0 ∧ r > 0 ∧ h > 0 ∧
  (1/2) * Real.pi * l^2 = 2 * Real.pi ∧  -- Area of semicircle is 2π
  2 * Real.pi * r = Real.pi * l ∧        -- Circumference of base equals arc length of semicircle
  h^2 + r^2 = l^2 →                      -- Pythagorean theorem
  (1/3) * Real.pi * r^2 * h = (Real.sqrt 3 / 3) * Real.pi := by
sorry


end NUMINAMATH_CALUDE_cone_volume_from_lateral_surface_l3878_387812


namespace NUMINAMATH_CALUDE_income_calculation_l3878_387858

/-- Calculates a person's income given the income to expenditure ratio and savings amount. -/
def calculate_income (income_ratio : ℕ) (expenditure_ratio : ℕ) (savings : ℕ) : ℕ :=
  (income_ratio * savings) / (income_ratio - expenditure_ratio)

/-- Theorem stating that given the specified conditions, the person's income is 10000. -/
theorem income_calculation (income_ratio : ℕ) (expenditure_ratio : ℕ) (savings : ℕ) 
  (h1 : income_ratio = 10)
  (h2 : expenditure_ratio = 7)
  (h3 : savings = 3000) :
  calculate_income income_ratio expenditure_ratio savings = 10000 := by
  sorry

end NUMINAMATH_CALUDE_income_calculation_l3878_387858


namespace NUMINAMATH_CALUDE_snow_leopard_arrangement_l3878_387801

theorem snow_leopard_arrangement (n : ℕ) (h : n = 9) : 
  2 * Nat.factorial (n - 2) = 10080 := by
  sorry

end NUMINAMATH_CALUDE_snow_leopard_arrangement_l3878_387801


namespace NUMINAMATH_CALUDE_total_renovation_time_is_79_5_l3878_387836

/-- Represents the renovation time for a house with specific room conditions. -/
def house_renovation_time (bedroom_time : ℝ) (bedroom_count : ℕ) (garden_time : ℝ) : ℝ :=
  let kitchen_time := 1.5 * bedroom_time
  let terrace_time := garden_time - 2
  let basement_time := 0.75 * kitchen_time
  let non_living_time := bedroom_time * bedroom_count + kitchen_time + garden_time + terrace_time + basement_time
  non_living_time + 2 * non_living_time

/-- Theorem stating that the total renovation time for the given house is 79.5 hours. -/
theorem total_renovation_time_is_79_5 :
  house_renovation_time 4 3 3 = 79.5 := by
  sorry

#eval house_renovation_time 4 3 3

end NUMINAMATH_CALUDE_total_renovation_time_is_79_5_l3878_387836


namespace NUMINAMATH_CALUDE_cubic_function_extrema_condition_l3878_387846

/-- A cubic function with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + (a + 6)*x + 1

/-- The derivative of f with respect to x -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + (a + 6)

/-- Theorem: If f has both a maximum and a minimum value, then a < -3 or a > 6 -/
theorem cubic_function_extrema_condition (a : ℝ) :
  (∃ (x_max x_min : ℝ), ∀ x, f a x ≤ f a x_max ∧ f a x_min ≤ f a x) →
  a < -3 ∨ a > 6 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_extrema_condition_l3878_387846


namespace NUMINAMATH_CALUDE_opposite_of_2023_l3878_387828

theorem opposite_of_2023 :
  ∃ x : ℤ, (2023 + x = 0) ∧ (x = -2023) :=
by sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l3878_387828


namespace NUMINAMATH_CALUDE_max_imaginary_part_of_roots_l3878_387805

open Complex

theorem max_imaginary_part_of_roots (z : ℂ) :
  z^6 - z^5 + z^4 - z^3 + z^2 - z + 1 = 0 →
  ∃ (φ : ℝ), -π/2 ≤ φ ∧ φ ≤ π/2 ∧
  (∀ (w : ℂ), w^6 - w^5 + w^4 - w^3 + w^2 - w + 1 = 0 →
    w.im ≤ Real.sin φ) ∧
  φ = (900 * π) / (7 * 180) :=
sorry

end NUMINAMATH_CALUDE_max_imaginary_part_of_roots_l3878_387805


namespace NUMINAMATH_CALUDE_cosine_product_in_special_sequence_l3878_387856

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)

theorem cosine_product_in_special_sequence (a₁ : ℝ) :
  let a := arithmetic_sequence a₁ (2 * Real.pi / 3)
  let S := {x | ∃ n : ℕ+, x = Real.cos (a n)}
  (∃ a b : ℝ, S = {a, b}) →
  ∃ a b : ℝ, S = {a, b} ∧ a * b = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_product_in_special_sequence_l3878_387856


namespace NUMINAMATH_CALUDE_acid_mixture_proof_l3878_387879

theorem acid_mixture_proof :
  let volume1 : ℝ := 4
  let concentration1 : ℝ := 0.60
  let volume2 : ℝ := 16
  let concentration2 : ℝ := 0.75
  let total_volume : ℝ := 20
  let final_concentration : ℝ := 0.72
  (volume1 * concentration1 + volume2 * concentration2) / total_volume = final_concentration ∧
  volume1 + volume2 = total_volume := by
sorry

end NUMINAMATH_CALUDE_acid_mixture_proof_l3878_387879


namespace NUMINAMATH_CALUDE_soda_cost_l3878_387803

theorem soda_cost (burger_cost soda_cost : ℕ) : 
  (3 * burger_cost + 2 * soda_cost = 450) →
  (2 * burger_cost + 3 * soda_cost = 480) →
  soda_cost = 108 := by
sorry

end NUMINAMATH_CALUDE_soda_cost_l3878_387803


namespace NUMINAMATH_CALUDE_box_interior_surface_area_l3878_387859

theorem box_interior_surface_area :
  let original_length : ℕ := 25
  let original_width : ℕ := 35
  let corner_size : ℕ := 7
  let original_area := original_length * original_width
  let corner_area := corner_size * corner_size
  let total_corner_area := 4 * corner_area
  let remaining_area := original_area - total_corner_area
  remaining_area = 679 := by sorry

end NUMINAMATH_CALUDE_box_interior_surface_area_l3878_387859


namespace NUMINAMATH_CALUDE_alchemerion_age_proof_l3878_387890

/-- Alchemerion's age in years -/
def alchemerion_age : ℕ := 277

/-- Alchemerion's son's age in years -/
def son_age : ℕ := alchemerion_age / 3

/-- Alchemerion's father's age in years -/
def father_age : ℕ := 2 * alchemerion_age + 40

/-- The sum of Alchemerion's, his son's, and his father's ages -/
def total_age : ℕ := alchemerion_age + son_age + father_age

theorem alchemerion_age_proof :
  alchemerion_age = 3 * son_age ∧
  father_age = 2 * alchemerion_age + 40 ∧
  total_age = 1240 →
  alchemerion_age = 277 := by
  sorry

end NUMINAMATH_CALUDE_alchemerion_age_proof_l3878_387890


namespace NUMINAMATH_CALUDE_sam_gave_29_cards_l3878_387898

/-- The number of new Pokemon cards Sam gave to Mary -/
def new_cards (initial : ℕ) (torn : ℕ) (final : ℕ) : ℕ :=
  final - (initial - torn)

/-- Proof that Sam gave Mary 29 new Pokemon cards -/
theorem sam_gave_29_cards : new_cards 33 6 56 = 29 := by
  sorry

end NUMINAMATH_CALUDE_sam_gave_29_cards_l3878_387898


namespace NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l3878_387833

theorem sum_of_absolute_coefficients (a a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (1 - x)^5 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  |a| + |a₁| + |a₂| + |a₃| + |a₄| + |a₅| = 32 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l3878_387833


namespace NUMINAMATH_CALUDE_iron_aluminum_weight_difference_l3878_387887

/-- The weight difference between two metal pieces -/
def weight_difference (iron_weight aluminum_weight : Float) : Float :=
  iron_weight - aluminum_weight

/-- Theorem stating the weight difference between iron and aluminum pieces -/
theorem iron_aluminum_weight_difference :
  let iron_weight : Float := 11.17
  let aluminum_weight : Float := 0.83
  weight_difference iron_weight aluminum_weight = 10.34 := by
  sorry

end NUMINAMATH_CALUDE_iron_aluminum_weight_difference_l3878_387887


namespace NUMINAMATH_CALUDE_stratified_sampling_female_count_l3878_387807

theorem stratified_sampling_female_count 
  (male_count : ℕ) 
  (female_count : ℕ) 
  (sample_size : ℕ) 
  (h1 : male_count = 810) 
  (h2 : female_count = 540) 
  (h3 : sample_size = 200) :
  (female_count * sample_size) / (male_count + female_count) = 80 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_female_count_l3878_387807


namespace NUMINAMATH_CALUDE_fibonacci_period_divisibility_l3878_387818

def is_extractable (p : ℕ) : Prop := ∃ x : ℕ, x * x ≡ 5 [MOD p]

def period_length (p : ℕ) : ℕ := sorry

theorem fibonacci_period_divisibility (p : ℕ) (hp : Prime p) (hp_neq : p ≠ 2 ∧ p ≠ 5) :
  (¬is_extractable p → (period_length p) ∣ (p + 1)) ∧
  (is_extractable p → (period_length p) ∣ (p - 1)) := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_period_divisibility_l3878_387818


namespace NUMINAMATH_CALUDE_sequence_problem_l3878_387824

def geometric_sequence (a : ℕ → ℝ) := ∀ n : ℕ, a (n + 1) / a n = a 2 / a 1

def arithmetic_sequence (b : ℕ → ℝ) := ∀ n : ℕ, b (n + 1) - b n = b 2 - b 1

theorem sequence_problem (a b : ℕ → ℝ) 
  (h1 : geometric_sequence a) 
  (h2 : arithmetic_sequence b) 
  (h3 : a 1 * a 6 * a 11 = -3 * Real.sqrt 3) 
  (h4 : b 1 + b 6 + b 11 = 7 * Real.pi) : 
  Real.tan ((b 3 + b 9) / (1 - a 4 * a 8)) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sequence_problem_l3878_387824


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3878_387853

def A : Set ℤ := {x | ∃ k : ℤ, x = 2 * k}
def B : Set ℤ := {-2, -1, 0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {-2, 0, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3878_387853


namespace NUMINAMATH_CALUDE_modulo_eleven_residue_l3878_387882

theorem modulo_eleven_residue : (308 + 6 * 44 + 8 * 165 + 3 * 18) % 11 = 10 := by
  sorry

end NUMINAMATH_CALUDE_modulo_eleven_residue_l3878_387882


namespace NUMINAMATH_CALUDE_software_contract_probability_l3878_387875

/-- Given probabilities for a computer company's contract scenarios, 
    prove the probability of not getting the software contract. -/
theorem software_contract_probability 
  (p_hardware : ℝ) 
  (p_at_least_one : ℝ) 
  (p_both : ℝ) 
  (h1 : p_hardware = 4/5)
  (h2 : p_at_least_one = 9/10)
  (h3 : p_both = 3/10) :
  1 - (p_at_least_one - p_hardware + p_both) = 3/5 :=
by sorry

end NUMINAMATH_CALUDE_software_contract_probability_l3878_387875


namespace NUMINAMATH_CALUDE_sine_cosine_sum_equals_sqrt3_over_2_l3878_387889

theorem sine_cosine_sum_equals_sqrt3_over_2 : 
  Real.sin (20 * π / 180) * Real.cos (40 * π / 180) + 
  Real.cos (20 * π / 180) * Real.sin (40 * π / 180) = 
  Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_sine_cosine_sum_equals_sqrt3_over_2_l3878_387889


namespace NUMINAMATH_CALUDE_matrix_transformation_l3878_387878

def matrix_A : Matrix (Fin 2) (Fin 2) ℤ := !![3, 6; 2, 7]
def matrix_B (x : ℤ) : Matrix (Fin 2) (Fin 2) ℤ := !![6, 2; 1, x]

theorem matrix_transformation (x : ℤ) : 
  Matrix.det matrix_A = Matrix.det (matrix_B x) → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_matrix_transformation_l3878_387878


namespace NUMINAMATH_CALUDE_eight_coin_stack_exists_fourteen_mm_stack_has_eight_coins_l3878_387809

/-- Represents the types of coins --/
inductive Coin
  | Penny
  | Nickel
  | Dime
  | Quarter

/-- Returns the thickness of a given coin in millimeters --/
def coinThickness (c : Coin) : ℚ :=
  match c with
  | Coin.Penny => 155/100
  | Coin.Nickel => 195/100
  | Coin.Dime => 135/100
  | Coin.Quarter => 175/100

/-- Represents a stack of coins --/
def CoinStack := List Coin

/-- Calculates the height of a coin stack in millimeters --/
def stackHeight (stack : CoinStack) : ℚ :=
  stack.foldl (fun acc c => acc + coinThickness c) 0

/-- Theorem: There exists a stack of 8 coins with a height of exactly 14 mm --/
theorem eight_coin_stack_exists : ∃ (stack : CoinStack), stackHeight stack = 14 ∧ stack.length = 8 := by
  sorry

/-- Theorem: Any stack of coins with a height of exactly 14 mm must contain 8 coins --/
theorem fourteen_mm_stack_has_eight_coins (stack : CoinStack) :
  stackHeight stack = 14 → stack.length = 8 := by
  sorry

end NUMINAMATH_CALUDE_eight_coin_stack_exists_fourteen_mm_stack_has_eight_coins_l3878_387809


namespace NUMINAMATH_CALUDE_prime_equivalence_l3878_387866

theorem prime_equivalence (k : ℕ) (h : ℕ) (n : ℕ) 
  (h_odd : Odd h) 
  (h_bound : h < 2^k) 
  (n_def : n = 2^k * h + 1) : 
  Nat.Prime n ↔ ∃ a : ℕ, a^((n-1)/2) % n = n - 1 := by
  sorry

end NUMINAMATH_CALUDE_prime_equivalence_l3878_387866


namespace NUMINAMATH_CALUDE_inequality_contradiction_l3878_387845

theorem inequality_contradiction (a b c d : ℝ) 
  (h1 : a < b) (h2 : b < c) (h3 : c < d) 
  (h4 : a / b = c / d) : 
  ¬((a + b) / (a - b) = (c + d) / (c - d)) := by
sorry

end NUMINAMATH_CALUDE_inequality_contradiction_l3878_387845


namespace NUMINAMATH_CALUDE_velocity_at_2s_l3878_387861

-- Define the displacement function
def S (t : ℝ) : ℝ := 10 * t - t^2

-- Define the velocity function as the derivative of displacement
def v (t : ℝ) : ℝ := 10 - 2 * t

-- Theorem statement
theorem velocity_at_2s :
  v 2 = 6 := by sorry

end NUMINAMATH_CALUDE_velocity_at_2s_l3878_387861


namespace NUMINAMATH_CALUDE_quadratic_positive_range_l3878_387888

def quadratic_function (a x : ℝ) : ℝ := a * x^2 - 2 * a * x + 3

theorem quadratic_positive_range (a : ℝ) :
  (∀ x : ℝ, 0 < x → x < 3 → quadratic_function a x > 0) ↔ 
  ((-1 ≤ a ∧ a < 0) ∨ (0 < a ∧ a < 3)) :=
sorry

end NUMINAMATH_CALUDE_quadratic_positive_range_l3878_387888


namespace NUMINAMATH_CALUDE_inequality_solution_l3878_387891

theorem inequality_solution (x : ℝ) (h : x ≠ 4) : (x^2 + 4) / ((x - 4)^2) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l3878_387891


namespace NUMINAMATH_CALUDE_min_value_theorem_l3878_387874

-- Define the function f
def f (a b c d x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

-- State the theorem
theorem min_value_theorem (a b c d : ℝ) (h1 : a < (2/3) * b) 
  (h2 : ∀ x y : ℝ, x < y → f a b c d x < f a b c d y) :
  ∃ m : ℝ, m = 1 ∧ ∀ k : ℝ, k = c / (2*b - 3*a) → m ≤ k :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3878_387874


namespace NUMINAMATH_CALUDE_jerry_money_duration_l3878_387873

/-- The number of weeks Jerry's money will last given his earnings and weekly spending -/
def weeks_money_lasts (lawn_earnings weed_eating_earnings weekly_spending : ℕ) : ℕ :=
  (lawn_earnings + weed_eating_earnings) / weekly_spending

/-- Theorem stating that Jerry's money will last 9 weeks -/
theorem jerry_money_duration :
  weeks_money_lasts 14 31 5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_jerry_money_duration_l3878_387873


namespace NUMINAMATH_CALUDE_smallest_r_minus_p_l3878_387867

theorem smallest_r_minus_p : ∃ (p q r : ℕ+),
  (p * q * r = 362880) ∧   -- 9! = 362880
  (p < q) ∧ (q < r) ∧
  ∀ (p' q' r' : ℕ+),
    (p' * q' * r' = 362880) →
    (p' < q') → (q' < r') →
    (r - p : ℤ) ≤ (r' - p' : ℤ) ∧
  (r - p : ℤ) = 219 := by
  sorry

end NUMINAMATH_CALUDE_smallest_r_minus_p_l3878_387867


namespace NUMINAMATH_CALUDE_ducks_theorem_l3878_387876

def ducks_remaining (initial : ℕ) : ℕ :=
  let after_first := initial - (initial / 4)
  let after_second := after_first - (after_first / 6)
  after_second - (after_second * 3 / 10)

theorem ducks_theorem : ducks_remaining 320 = 140 := by
  sorry

end NUMINAMATH_CALUDE_ducks_theorem_l3878_387876


namespace NUMINAMATH_CALUDE_mary_pizza_order_l3878_387863

def large_pizza_slices : ℕ := 8
def slices_eaten : ℕ := 7
def slices_remaining : ℕ := 9

theorem mary_pizza_order : 
  ∃ (pizzas_ordered : ℕ), 
    pizzas_ordered * large_pizza_slices = slices_eaten + slices_remaining ∧ 
    pizzas_ordered = 2 := by
  sorry

end NUMINAMATH_CALUDE_mary_pizza_order_l3878_387863


namespace NUMINAMATH_CALUDE_rectangle_area_l3878_387894

/-- A rectangle with three congruent circles inside -/
structure RectangleWithCircles where
  -- The length of the rectangle
  length : ℝ
  -- The width of the rectangle
  width : ℝ
  -- The diameter of each circle
  circle_diameter : ℝ
  -- The circles are congruent
  circles_congruent : True
  -- Each circle is tangent to two sides of the rectangle
  circles_tangent : True
  -- The circle centered at F is tangent to sides JK and LM
  circle_f_tangent : True
  -- The diameter of circle F is 5
  circle_f_diameter : circle_diameter = 5

/-- The area of the rectangle JKLM is 50 -/
theorem rectangle_area (r : RectangleWithCircles) : r.length * r.width = 50 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3878_387894


namespace NUMINAMATH_CALUDE_find_N_l3878_387895

theorem find_N : ∃ N : ℕ, (10 + 11 + 12 + 13) / 4 = (1000 + 1001 + 1002 + 1003) / N ∧ N = 348 := by
  sorry

end NUMINAMATH_CALUDE_find_N_l3878_387895


namespace NUMINAMATH_CALUDE_x_value_when_y_is_5_l3878_387862

-- Define the constant ratio
def k : ℚ := (5 * 3 - 6) / (2 * 2 + 10)

-- Define the relationship between x and y
def relation (x y : ℚ) : Prop := (5 * x - 6) / (2 * y + 10) = k

-- State the theorem
theorem x_value_when_y_is_5 :
  ∀ x : ℚ, relation x 2 → relation 3 2 → relation x 5 → x = 53 / 14 :=
sorry

end NUMINAMATH_CALUDE_x_value_when_y_is_5_l3878_387862


namespace NUMINAMATH_CALUDE_batch_size_proof_l3878_387826

theorem batch_size_proof (x : ℕ) (N : ℕ) :
  (20 * (x - 1) = N) →                   -- Condition 1
  (∃ r : ℕ, r = 20) →                    -- Original rate
  ((25 * (x - 7)) = N - 80) →            -- Condition 2 (after rate increase)
  (x = 14) →                             -- Derived from solution
  N = 280 :=                             -- Conclusion to prove
by sorry

end NUMINAMATH_CALUDE_batch_size_proof_l3878_387826


namespace NUMINAMATH_CALUDE_triangle_third_vertex_l3878_387817

/-- Given an obtuse triangle with vertices at (8, 6), (0, 0), and (x, 0),
    if the area of the triangle is 48 square units, then x = 16 or x = -16 -/
theorem triangle_third_vertex (x : ℝ) : 
  let v1 : ℝ × ℝ := (8, 6)
  let v2 : ℝ × ℝ := (0, 0)
  let v3 : ℝ × ℝ := (x, 0)
  let triangle_area := (1/2 : ℝ) * |v1.1 * (v2.2 - v3.2) + v2.1 * (v3.2 - v1.2) + v3.1 * (v1.2 - v2.2)|
  (triangle_area = 48) → (x = 16 ∨ x = -16) :=
by sorry

end NUMINAMATH_CALUDE_triangle_third_vertex_l3878_387817


namespace NUMINAMATH_CALUDE_function_characterization_l3878_387841

theorem function_characterization
  (f : ℤ → ℤ)
  (h : ∀ m n : ℤ, f m + f n = max (f (m + n)) (f (m - n))) :
  ∃ k : ℕ, ∀ x : ℤ, f x = k * |x| :=
sorry

end NUMINAMATH_CALUDE_function_characterization_l3878_387841


namespace NUMINAMATH_CALUDE_prob_at_least_one_woman_l3878_387892

theorem prob_at_least_one_woman (men women selected : ℕ) :
  men = 9 →
  women = 5 →
  selected = 3 →
  (1 - (Nat.choose men selected) / (Nat.choose (men + women) selected) : ℚ) = 23/30 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_woman_l3878_387892


namespace NUMINAMATH_CALUDE_circle_arcs_angle_sum_l3878_387899

theorem circle_arcs_angle_sum (n : ℕ) (x y : ℝ) : 
  n = 18 → 
  x = 3 * (360 / n) / 2 →
  y = 5 * (360 / n) / 2 →
  x + y = 80 := by
  sorry

end NUMINAMATH_CALUDE_circle_arcs_angle_sum_l3878_387899


namespace NUMINAMATH_CALUDE_correct_calculation_l3878_387849

theorem correct_calculation (x : ℤ) : 954 - x = 468 → 954 + x = 1440 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l3878_387849


namespace NUMINAMATH_CALUDE_sin_five_zeros_l3878_387850

theorem sin_five_zeros (f : ℝ → ℝ) (ω : ℝ) :
  ω > 0 →
  (∀ x, f x = Real.sin (ω * x)) →
  (∃! z : Finset ℝ, z.card = 5 ∧ (∀ x ∈ z, x ∈ Set.Icc 0 (3 * Real.pi) ∧ f x = 0)) →
  ω ∈ Set.Icc (4 / 3) (5 / 3) :=
sorry

end NUMINAMATH_CALUDE_sin_five_zeros_l3878_387850


namespace NUMINAMATH_CALUDE_complex_location_l3878_387823

theorem complex_location (z : ℂ) (h : (z - 3) * (2 - Complex.I) = 5) : 
  0 < z.re ∧ 0 < z.im := by
  sorry

end NUMINAMATH_CALUDE_complex_location_l3878_387823


namespace NUMINAMATH_CALUDE_actual_annual_yield_actual_annual_yield_approx_l3878_387802

/-- Calculates the actual annual yield for a one-year term deposit with varying interest rates and a closing fee. -/
theorem actual_annual_yield (P : ℝ) : ℝ :=
  let first_quarter_rate := 0.12 / 4
  let second_quarter_rate := 0.08 / 4
  let third_semester_rate := 0.06 / 2
  let closing_fee_rate := 0.01
  let final_amount := P * (1 + first_quarter_rate) * (1 + second_quarter_rate) * (1 + third_semester_rate)
  let effective_final_amount := final_amount - (P * closing_fee_rate)
  (effective_final_amount / P) - 1

/-- The actual annual yield is approximately 7.2118% -/
theorem actual_annual_yield_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.00001 ∧ ∀ (P : ℝ), P > 0 → |actual_annual_yield P - 0.072118| < ε :=
sorry

end NUMINAMATH_CALUDE_actual_annual_yield_actual_annual_yield_approx_l3878_387802


namespace NUMINAMATH_CALUDE_simultaneous_equations_solution_l3878_387843

theorem simultaneous_equations_solution :
  ∀ x y : ℝ,
  (3 * x^2 + x * y - 2 * y^2 = -5 ∧ x^2 + 2 * x * y + y^2 = 1) ↔
  ((x = 3/5 ∧ y = -8/5) ∨ (x = -3/5 ∧ y = 8/5)) :=
by sorry

end NUMINAMATH_CALUDE_simultaneous_equations_solution_l3878_387843


namespace NUMINAMATH_CALUDE_term_position_98_l3878_387868

/-- The sequence defined by a_n = n^2 / (n^2 + 1) -/
def a (n : ℕ) : ℚ := n^2 / (n^2 + 1)

/-- The theorem stating that 0.98 occurs at position 7 in the sequence -/
theorem term_position_98 : a 7 = 98/100 := by
  sorry

end NUMINAMATH_CALUDE_term_position_98_l3878_387868


namespace NUMINAMATH_CALUDE_square_area_error_l3878_387897

theorem square_area_error (s : ℝ) (h : s > 0) :
  let measured_side := s * (1 + 0.02)
  let actual_area := s^2
  let calculated_area := measured_side^2
  let area_error := (calculated_area - actual_area) / actual_area
  area_error = 0.0404 := by
sorry

end NUMINAMATH_CALUDE_square_area_error_l3878_387897


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l3878_387800

def a : ℝ × ℝ := (1, 0)
def b : ℝ × ℝ := (0, 1)

theorem perpendicular_vectors :
  let v1 := (2 * a.1 + b.1, 2 * a.2 + b.2)
  let v2 := (a.1 - 2 * b.1, a.2 - 2 * b.2)
  v1.1 * v2.1 + v1.2 * v2.2 = 0 := by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l3878_387800


namespace NUMINAMATH_CALUDE_tortoise_age_problem_l3878_387881

theorem tortoise_age_problem (tailor_age tortoise_age tree_age : ℕ) : 
  tailor_age + tortoise_age + tree_age = 264 →
  tailor_age = 4 * (tailor_age - tortoise_age) →
  tortoise_age = 7 * (tortoise_age - tree_age) →
  tortoise_age = 77 := by
sorry

end NUMINAMATH_CALUDE_tortoise_age_problem_l3878_387881


namespace NUMINAMATH_CALUDE_initial_money_calculation_l3878_387854

theorem initial_money_calculation (initial_money : ℚ) : 
  (2 / 5 : ℚ) * initial_money = 200 → initial_money = 500 := by
  sorry

#check initial_money_calculation

end NUMINAMATH_CALUDE_initial_money_calculation_l3878_387854


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3878_387848

theorem inequality_solution_set :
  {x : ℝ | 5 - x^2 > 4*x} = Set.Ioo (-5 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3878_387848


namespace NUMINAMATH_CALUDE_work_completion_men_count_first_group_size_l3878_387847

theorem work_completion_men_count : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun first_group_days second_group_men second_group_days result =>
    first_group_days * result = second_group_men * second_group_days

theorem first_group_size :
  ∃ (m : ℕ), work_completion_men_count 80 20 40 m ∧ m = 10 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_men_count_first_group_size_l3878_387847


namespace NUMINAMATH_CALUDE_c_value_l3878_387838

theorem c_value (x y : ℝ) (h : 2 * x + 5 * y = 3) :
  let c := Real.sqrt ((4 ^ (x + 1/2)) * (32 ^ y))
  c = 4 := by
sorry

end NUMINAMATH_CALUDE_c_value_l3878_387838


namespace NUMINAMATH_CALUDE_smallest_other_integer_l3878_387816

theorem smallest_other_integer (m n x : ℕ) : 
  m = 30 → 
  x > 0 → 
  Nat.gcd m n = x + 3 → 
  Nat.lcm m n = x * (x + 3) → 
  n ≥ 70 ∧ ∃ (n' : ℕ), n' = 70 ∧ 
    Nat.gcd m n' = x + 3 ∧ 
    Nat.lcm m n' = x * (x + 3) := by
  sorry

end NUMINAMATH_CALUDE_smallest_other_integer_l3878_387816


namespace NUMINAMATH_CALUDE_power_of_power_three_squared_four_l3878_387865

theorem power_of_power_three_squared_four : (3^2)^4 = 6561 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_three_squared_four_l3878_387865


namespace NUMINAMATH_CALUDE_jen_addition_problem_l3878_387884

/-- Rounds a natural number to the nearest hundred. -/
def roundToNearestHundred (n : ℕ) : ℕ :=
  (n + 50) / 100 * 100

/-- The problem statement -/
theorem jen_addition_problem :
  roundToNearestHundred (178 + 269) = 400 := by
  sorry

end NUMINAMATH_CALUDE_jen_addition_problem_l3878_387884


namespace NUMINAMATH_CALUDE_saturday_practice_hours_l3878_387885

/-- Given a person's practice schedule, calculate the hours practiced on Saturdays -/
theorem saturday_practice_hours 
  (weekday_hours : ℕ) 
  (total_weeks : ℕ) 
  (total_practice_hours : ℕ) 
  (h1 : weekday_hours = 3)
  (h2 : total_weeks = 3)
  (h3 : total_practice_hours = 60) :
  (total_practice_hours - weekday_hours * 5 * total_weeks) / total_weeks = 5 := by
  sorry

#check saturday_practice_hours

end NUMINAMATH_CALUDE_saturday_practice_hours_l3878_387885


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3878_387837

theorem polynomial_simplification (r : ℝ) : 
  (2 * r^3 + 5 * r^2 + 4 * r - 3) - (r^3 + 4 * r^2 + 6 * r - 8) = r^3 + r^2 - 2 * r + 5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3878_387837


namespace NUMINAMATH_CALUDE_largest_equal_division_l3878_387855

theorem largest_equal_division (tim_sweets peter_sweets : ℕ) 
  (h1 : tim_sweets = 36) (h2 : peter_sweets = 44) : 
  Nat.gcd tim_sweets peter_sweets = 4 := by
  sorry

end NUMINAMATH_CALUDE_largest_equal_division_l3878_387855


namespace NUMINAMATH_CALUDE_circles_and_line_properties_l3878_387840

-- Define Circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 5 = 0

-- Define Circle D
def circle_D (x y : ℝ) : Prop := (x - 5)^2 + (y - 4)^2 = 4

-- Define the tangent line l
def line_l (x y : ℝ) : Prop := x = 5 ∨ 7*x - 24*y + 61 = 0

-- Theorem statement
theorem circles_and_line_properties :
  -- Part 1: Circles C and D are externally tangent
  (∃ (x y : ℝ), circle_C x y ∧ circle_D x y) ∧
  -- The distance between centers is equal to the sum of radii
  ((2 - 5)^2 + (0 - 4)^2 : ℝ) = (3 + 2)^2 ∧
  -- Part 2: Line l is tangent to Circle C and passes through (5,4)
  (∀ (x y : ℝ), line_l x y → 
    -- Line passes through (5,4)
    (x = 5 ∧ y = 4 ∨ 7*5 - 24*4 + 61 = 0) ∧
    -- Line is tangent to Circle C (distance from center to line is equal to radius)
    ((2*7 + 0*(-24) - 61)^2 / (7^2 + (-24)^2) : ℝ) = 3^2) :=
sorry

end NUMINAMATH_CALUDE_circles_and_line_properties_l3878_387840


namespace NUMINAMATH_CALUDE_cube_root_properties_l3878_387814

theorem cube_root_properties :
  let n : ℕ := 59319
  let a : ℕ := 6859
  let b : ℕ := 19683
  let c : ℕ := 110592
  ∃ (x y z : ℕ),
    (10 ≤ x ∧ x < 100) ∧
    x^3 = n ∧
    x = 39 ∧
    y^3 = a ∧ y = 19 ∧
    z^3 = b ∧ z = 27 ∧
    (∃ w : ℕ, w^3 = c ∧ w = 48) :=
by sorry

end NUMINAMATH_CALUDE_cube_root_properties_l3878_387814


namespace NUMINAMATH_CALUDE_rectangle_to_hexagon_area_l3878_387831

/-- Given a rectangle with sides of length a and 36, prove that when transformed into a hexagon
    with parallel sides of length a separated by 24, and the hexagon has the same area as the
    original rectangle, then a² = 720. -/
theorem rectangle_to_hexagon_area (a : ℝ) : 
  (0 < a) →
  (24 * a + 30 * Real.sqrt (a^2 - 36) = 36 * a) →
  a^2 = 720 := by
sorry

end NUMINAMATH_CALUDE_rectangle_to_hexagon_area_l3878_387831


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l3878_387860

/-- Two vectors in ℝ² are perpendicular if their dot product is zero -/
def perpendicular (a b : ℝ × ℝ) : Prop := a.1 * b.1 + a.2 * b.2 = 0

/-- Given two vectors a and b in ℝ², where a = (1,3) and b = (x,-1),
    if a is perpendicular to b, then x = 3 -/
theorem perpendicular_vectors (x : ℝ) : 
  perpendicular (1, 3) (x, -1) → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l3878_387860


namespace NUMINAMATH_CALUDE_work_scaling_l3878_387896

/-- Given that 3 people can do 3 times of a particular work in 3 days,
    prove that 9 people can do 9 times of that particular work in 3 days. -/
theorem work_scaling (work_rate : ℕ → ℕ → ℝ → ℝ) :
  work_rate 3 3 3 = 1 →
  work_rate 9 9 3 = 1 := by
sorry

end NUMINAMATH_CALUDE_work_scaling_l3878_387896


namespace NUMINAMATH_CALUDE_angle_calculation_l3878_387825

/-- Given an angle α with its vertex at the origin, its initial side coinciding with
    the non-negative half-axis of the x-axis, and a point P(-2, -1) on its terminal side,
    prove that 2cos²α - sin(π - 2α) = 4/5 -/
theorem angle_calculation (α : ℝ) :
  (∃ (P : ℝ × ℝ), P = (-2, -1) ∧
    P.1 = Real.cos α * Real.sqrt 5 ∧
    P.2 = Real.sin α * Real.sqrt 5) →
  2 * (Real.cos α)^2 - Real.sin (π - 2 * α) = 4/5 :=
by sorry

end NUMINAMATH_CALUDE_angle_calculation_l3878_387825


namespace NUMINAMATH_CALUDE_triangle_side_length_l3878_387880

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  sinB : ℝ
  area : ℝ

-- Define the conditions
def isValidTriangle (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b

def isArithmeticSequence (t : Triangle) : Prop :=
  2 * t.b = t.a + t.c

def hasSinB (t : Triangle) : Prop :=
  t.sinB = 4/5

def hasArea (t : Triangle) : Prop :=
  t.area = 3/2

-- Theorem statement
theorem triangle_side_length (t : Triangle) 
  (h1 : isValidTriangle t)
  (h2 : isArithmeticSequence t)
  (h3 : hasSinB t)
  (h4 : hasArea t) :
  t.b = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3878_387880


namespace NUMINAMATH_CALUDE_tensor_A_equals_result_l3878_387835

def A : Set ℕ := {0, 2, 3}

def tensor_operation (S : Set ℕ) : Set ℕ :=
  {x | ∃ a b, a ∈ S ∧ b ∈ S ∧ x = a + b}

theorem tensor_A_equals_result : tensor_operation A = {0, 2, 3, 4, 5, 6} := by
  sorry

end NUMINAMATH_CALUDE_tensor_A_equals_result_l3878_387835


namespace NUMINAMATH_CALUDE_profit_formula_l3878_387810

-- Define variables
variable (C S P p n : ℝ)

-- Define the conditions
def condition1 : Prop := P = p * ((C + S) / 2)
def condition2 : Prop := P = S / n - C

-- Theorem statement
theorem profit_formula 
  (h1 : condition1 C S P p)
  (h2 : condition2 C S P n)
  : P = (S * (2 * n * p + 2 * p - n)) / (n * (2 * p + n)) :=
by sorry

end NUMINAMATH_CALUDE_profit_formula_l3878_387810


namespace NUMINAMATH_CALUDE_sqrt_81_equals_3_squared_l3878_387821

theorem sqrt_81_equals_3_squared : Real.sqrt 81 = 3^2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_81_equals_3_squared_l3878_387821


namespace NUMINAMATH_CALUDE_min_value_sum_squares_l3878_387844

theorem min_value_sum_squares (a b : ℝ) : 
  a > 0 → b > 0 → a ≠ b → a^2 - 2015*a = b^2 - 2015*b → 
  ∀ x y : ℝ, x > 0 → y > 0 → x ≠ y → x^2 - 2015*x = y^2 - 2015*y → 
  a^2 + b^2 ≤ x^2 + y^2 ∧ a^2 + b^2 = 2015^2 / 2 := by
sorry

end NUMINAMATH_CALUDE_min_value_sum_squares_l3878_387844


namespace NUMINAMATH_CALUDE_parabola_translation_l3878_387829

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (h v : ℝ) : Parabola :=
  { a := p.a
    b := p.b - 2 * p.a * h
    c := p.c + p.a * h^2 - p.b * h - v }

theorem parabola_translation (x y : ℝ) :
  let original := Parabola.mk (-1) (-2) 0
  let translated := translate (translate original 2 0) 0 3
  y = -(x * (x + 2)) → y = -(x - 1)^2 - 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_translation_l3878_387829


namespace NUMINAMATH_CALUDE_square_cards_puzzle_l3878_387822

theorem square_cards_puzzle (n : ℕ) (h : n > 0) (eq : n^2 + 36 = (n + 1)^2 + 3) :
  n^2 + 36 = 292 := by
  sorry

end NUMINAMATH_CALUDE_square_cards_puzzle_l3878_387822


namespace NUMINAMATH_CALUDE_one_zero_of_sin_log_l3878_387813

open Real

noncomputable def f (x : ℝ) : ℝ := sin (log x)

theorem one_zero_of_sin_log (h : ∀ x, 1 < x → x < exp π → f x = 0 → x = exp π) :
  ∃! x, 1 < x ∧ x < exp π ∧ f x = 0 :=
sorry

end NUMINAMATH_CALUDE_one_zero_of_sin_log_l3878_387813


namespace NUMINAMATH_CALUDE_complement_determines_set_l3878_387877

def U : Set Nat := {0, 1, 2, 3}

theorem complement_determines_set (A : Set Nat) 
  (h1 : U = {0, 1, 2, 3})
  (h2 : (U \ A) = {2}) : 
  A = {0, 1, 3} := by
sorry

end NUMINAMATH_CALUDE_complement_determines_set_l3878_387877


namespace NUMINAMATH_CALUDE_xyz_value_l3878_387864

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 27)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 9) : 
  x * y * z = 6 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l3878_387864


namespace NUMINAMATH_CALUDE_special_polynomial_sum_l3878_387893

/-- A monic polynomial of degree 4 satisfying specific conditions -/
def SpecialPolynomial (p : ℝ → ℝ) : Prop :=
  (∀ x, ∃ a b c d : ℝ, p x = x^4 + a*x^3 + b*x^2 + c*x + d) ∧
  p 1 = 20 ∧ p 2 = 40 ∧ p 3 = 60

/-- The sum of p(0) and p(4) for a special polynomial p is 92 -/
theorem special_polynomial_sum (p : ℝ → ℝ) (h : SpecialPolynomial p) : 
  p 0 + p 4 = 92 := by
  sorry

end NUMINAMATH_CALUDE_special_polynomial_sum_l3878_387893


namespace NUMINAMATH_CALUDE_more_stable_lower_variance_l3878_387883

/-- Represents an athlete's assessment scores -/
structure AthleteScores where
  variance : ℝ
  assessmentCount : ℕ

/-- Defines the stability of an athlete's scores based on variance -/
def moreStable (a b : AthleteScores) : Prop :=
  a.variance < b.variance

/-- Theorem: Given two athletes with the same average score but different variances,
    the athlete with lower variance has more stable scores -/
theorem more_stable_lower_variance 
  (athleteA athleteB : AthleteScores)
  (hCount : athleteA.assessmentCount = athleteB.assessmentCount)
  (hCountPos : athleteA.assessmentCount > 0)
  (hVarA : athleteA.variance = 1.43)
  (hVarB : athleteB.variance = 0.82) :
  moreStable athleteB athleteA := by
  sorry

#check more_stable_lower_variance

end NUMINAMATH_CALUDE_more_stable_lower_variance_l3878_387883


namespace NUMINAMATH_CALUDE_min_value_of_3a_plus_b_l3878_387820

theorem min_value_of_3a_plus_b (a b : ℝ) (h : 16 * a^2 + 2 * a + 8 * a * b + b^2 - 1 = 0) :
  ∃ (m : ℝ), m = 3 * a + b ∧ m ≥ -1 ∧ ∀ (x : ℝ), (∃ (a' b' : ℝ), x = 3 * a' + b' ∧ 16 * a'^2 + 2 * a' + 8 * a' * b' + b'^2 - 1 = 0) → x ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_of_3a_plus_b_l3878_387820


namespace NUMINAMATH_CALUDE_inequality_proof_l3878_387886

theorem inequality_proof (x y : ℝ) (h : x^4 + y^4 ≤ 1) : x^6 - y^6 + 2*y^3 < π/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3878_387886


namespace NUMINAMATH_CALUDE_product_is_even_l3878_387806

theorem product_is_even (a b c : ℤ) : 
  ∃ k : ℤ, (7 * a + b - 2 * c + 1) * (3 * a - 5 * b + 4 * c + 10) = 2 * k := by
sorry

end NUMINAMATH_CALUDE_product_is_even_l3878_387806


namespace NUMINAMATH_CALUDE_quadratic_inequality_roots_l3878_387832

/-- Given a quadratic function f(x) = -2x^2 + cx - 8, 
    where f(x) < 0 only when x ∈ (-∞, 2) ∪ (6, ∞),
    prove that c = 16 -/
theorem quadratic_inequality_roots (c : ℝ) : 
  (∀ x : ℝ, -2 * x^2 + c * x - 8 < 0 ↔ x < 2 ∨ x > 6) → 
  c = 16 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_roots_l3878_387832


namespace NUMINAMATH_CALUDE_product_of_fractions_l3878_387811

theorem product_of_fractions :
  (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l3878_387811
