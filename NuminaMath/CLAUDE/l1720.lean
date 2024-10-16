import Mathlib

namespace NUMINAMATH_CALUDE_alice_ice_cream_l1720_172099

/-- The number of pints of ice cream Alice had on Wednesday -/
def ice_cream_pints : ℕ → ℕ
  | 0 => 4  -- Sunday
  | 1 => 3 * ice_cream_pints 0  -- Monday
  | 2 => ice_cream_pints 1 / 3  -- Tuesday
  | 3 => ice_cream_pints 0 + ice_cream_pints 1 + ice_cream_pints 2 - ice_cream_pints 2 / 2  -- Wednesday
  | _ => 0  -- Other days (not relevant to the problem)

theorem alice_ice_cream : ice_cream_pints 3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_alice_ice_cream_l1720_172099


namespace NUMINAMATH_CALUDE_common_chord_of_circles_l1720_172069

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - 12*x - 2*y - 13 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 + 12*x + 16*y - 25 = 0

-- Define the common chord
def common_chord (x y : ℝ) : Prop := 4*x + 3*y - 2 = 0

-- Theorem statement
theorem common_chord_of_circles :
  ∀ x y : ℝ, (C₁ x y ∧ C₂ x y) → common_chord x y :=
by sorry

end NUMINAMATH_CALUDE_common_chord_of_circles_l1720_172069


namespace NUMINAMATH_CALUDE_theft_loss_calculation_l1720_172020

/-- Represents the percentage of profit taken by the shopkeeper -/
def profit_percentage : ℝ := 10

/-- Represents the overall loss percentage -/
def overall_loss_percentage : ℝ := 23

/-- Represents the percentage of goods lost during theft -/
def theft_loss_percentage : ℝ := 30

/-- Theorem stating the relationship between profit, overall loss, and theft loss -/
theorem theft_loss_calculation (cost : ℝ) (cost_positive : cost > 0) :
  let selling_price := cost * (1 + profit_percentage / 100)
  let actual_revenue := cost * (1 - overall_loss_percentage / 100)
  selling_price * (1 - theft_loss_percentage / 100) = actual_revenue :=
sorry

end NUMINAMATH_CALUDE_theft_loss_calculation_l1720_172020


namespace NUMINAMATH_CALUDE_cross_section_area_cross_section_area_is_14_l1720_172093

/-- Regular triangular prism with cross-section --/
structure Prism where
  a : ℝ  -- side length of the base
  S_ABC : ℝ  -- area of the base
  base_area_eq : S_ABC = a^2 * Real.sqrt 3 / 4
  D_midpoint : ℝ  -- D is midpoint of AB
  D_midpoint_eq : D_midpoint = a / 2
  K_on_BC : ℝ  -- distance BK
  K_on_BC_eq : K_on_BC = 3 * a / 4
  M_on_AC1 : ℝ  -- height of the prism
  N_on_A1B1 : ℝ  -- distance BG (projection of N)
  N_on_A1B1_eq : N_on_A1B1 = a / 6

/-- Theorem: The area of the cross-section is 14 --/
theorem cross_section_area (p : Prism) : ℝ :=
  let S_np := p.S_ABC * (3/8 - 1/24)  -- area of projection
  let cos_alpha := 1 / Real.sqrt 3
  S_np / cos_alpha

/-- Main theorem: The area of the cross-section is equal to 14 --/
theorem cross_section_area_is_14 (p : Prism) : cross_section_area p = 14 := by
  sorry

end NUMINAMATH_CALUDE_cross_section_area_cross_section_area_is_14_l1720_172093


namespace NUMINAMATH_CALUDE_bushes_needed_bushes_needed_proof_l1720_172007

/-- The number of containers of blueberries yielded by each bush -/
def containers_per_bush : ℕ := 10

/-- The number of containers of blueberries that can be traded for zucchinis -/
def containers_for_trade : ℕ := 6

/-- The number of zucchinis received in trade for containers_for_trade -/
def zucchinis_from_trade : ℕ := 3

/-- The target number of zucchinis -/
def target_zucchinis : ℕ := 60

/-- Theorem: The number of bushes needed to obtain the target number of zucchinis -/
theorem bushes_needed : ℕ := 12

/-- Proof that bushes_needed is correct -/
theorem bushes_needed_proof : 
  bushes_needed * containers_per_bush * zucchinis_from_trade = 
  target_zucchinis * containers_for_trade :=
by sorry

end NUMINAMATH_CALUDE_bushes_needed_bushes_needed_proof_l1720_172007


namespace NUMINAMATH_CALUDE_ac_squared_greater_implies_a_greater_l1720_172023

theorem ac_squared_greater_implies_a_greater (a b c : ℝ) : a * c^2 > b * c^2 → a > b := by
  sorry

end NUMINAMATH_CALUDE_ac_squared_greater_implies_a_greater_l1720_172023


namespace NUMINAMATH_CALUDE_circle_max_area_center_l1720_172010

/-- A circle with equation x^2 + y^2 + kx + 2y + k^2 = 0 -/
def Circle (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 + k * p.1 + 2 * p.2 + k^2 = 0}

/-- The center of a circle -/
def center (c : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

/-- The area of a circle -/
def area (c : Set (ℝ × ℝ)) : ℝ := sorry

/-- Theorem: The center of the circle is (0, -1) when its area is maximized -/
theorem circle_max_area_center (k : ℝ) :
  (∀ k' : ℝ, area (Circle k') ≤ area (Circle k)) →
  center (Circle k) = (0, -1) := by sorry

end NUMINAMATH_CALUDE_circle_max_area_center_l1720_172010


namespace NUMINAMATH_CALUDE_ratio_sum_theorem_l1720_172083

theorem ratio_sum_theorem (a b c : ℝ) (h : a / 3 = b / 4 ∧ b / 4 = c / 5) :
  (a + b + c) / c = 12 / 5 := by sorry

end NUMINAMATH_CALUDE_ratio_sum_theorem_l1720_172083


namespace NUMINAMATH_CALUDE_image_of_negative_two_three_preimage_of_two_negative_three_l1720_172085

-- Define the mapping f
def f (x y : ℝ) : ℝ × ℝ := (x + y, x * y)

-- Theorem for the image of (-2, 3)
theorem image_of_negative_two_three :
  f (-2) 3 = (1, -6) := by sorry

-- Theorem for the preimage of (2, -3)
theorem preimage_of_two_negative_three :
  {p : ℝ × ℝ | f p.1 p.2 = (2, -3)} = {(-1, 3), (3, -1)} := by sorry

end NUMINAMATH_CALUDE_image_of_negative_two_three_preimage_of_two_negative_three_l1720_172085


namespace NUMINAMATH_CALUDE_trinomial_zeros_l1720_172071

theorem trinomial_zeros (a b : ℝ) (ha : a > 4) (hb : b > 4) :
  (a^2 - 4*b > 0) ∨ (b^2 - 4*a > 0) := by sorry

end NUMINAMATH_CALUDE_trinomial_zeros_l1720_172071


namespace NUMINAMATH_CALUDE_loaves_baked_l1720_172073

def flour_available : ℝ := 5
def flour_per_loaf : ℝ := 2.5

theorem loaves_baked : ⌊flour_available / flour_per_loaf⌋ = 2 := by
  sorry

end NUMINAMATH_CALUDE_loaves_baked_l1720_172073


namespace NUMINAMATH_CALUDE_amanda_purchase_cost_l1720_172016

def dress_price : ℚ := 50
def shoes_price : ℚ := 75
def dress_discount : ℚ := 0.30
def shoes_discount : ℚ := 0.25
def tax_rate : ℚ := 0.05

def total_cost : ℚ :=
  let dress_discounted := dress_price * (1 - dress_discount)
  let shoes_discounted := shoes_price * (1 - shoes_discount)
  let subtotal := dress_discounted + shoes_discounted
  let tax := subtotal * tax_rate
  subtotal + tax

theorem amanda_purchase_cost : total_cost = 95.81 := by
  sorry

end NUMINAMATH_CALUDE_amanda_purchase_cost_l1720_172016


namespace NUMINAMATH_CALUDE_floor_sqrt_50_squared_l1720_172045

theorem floor_sqrt_50_squared : ⌊Real.sqrt 50⌋^2 = 49 := by sorry

end NUMINAMATH_CALUDE_floor_sqrt_50_squared_l1720_172045


namespace NUMINAMATH_CALUDE_tricycle_count_l1720_172002

theorem tricycle_count (total_children : ℕ) (total_wheels : ℕ) 
  (h1 : total_children = 10) 
  (h2 : total_wheels = 26) : ∃ (bicycles tricycles : ℕ),
  bicycles + tricycles = total_children ∧ 
  2 * bicycles + 3 * tricycles = total_wheels ∧
  tricycles = 6 := by
  sorry

end NUMINAMATH_CALUDE_tricycle_count_l1720_172002


namespace NUMINAMATH_CALUDE_heptagon_diagonals_l1720_172060

-- Define a heptagon
def Heptagon : Nat := 7

-- Define the formula for the number of diagonals in a polygon
def numDiagonals (n : Nat) : Nat := n * (n - 3) / 2

-- Theorem: The number of diagonals in a heptagon is 14
theorem heptagon_diagonals : numDiagonals Heptagon = 14 := by
  sorry

end NUMINAMATH_CALUDE_heptagon_diagonals_l1720_172060


namespace NUMINAMATH_CALUDE_smallest_lcm_with_gcd_five_l1720_172040

theorem smallest_lcm_with_gcd_five (k l : ℕ) : 
  1000 ≤ k ∧ k < 10000 ∧ 
  1000 ≤ l ∧ l < 10000 ∧ 
  Nat.gcd k l = 5 →
  201000 ≤ Nat.lcm k l ∧ 
  ∃ (k' l' : ℕ), 1000 ≤ k' ∧ k' < 10000 ∧ 
                 1000 ≤ l' ∧ l' < 10000 ∧ 
                 Nat.gcd k' l' = 5 ∧ 
                 Nat.lcm k' l' = 201000 :=
by sorry

end NUMINAMATH_CALUDE_smallest_lcm_with_gcd_five_l1720_172040


namespace NUMINAMATH_CALUDE_range_of_expression_l1720_172084

theorem range_of_expression (α β : ℝ) 
  (h_α : α ∈ Set.Ioo 0 (Real.pi / 2))
  (h_β : β ∈ Set.Icc 0 (Real.pi / 2)) :
  ∃ (x : ℝ), x ∈ Set.Ioo (-Real.pi / 6) Real.pi ∧
  ∃ (α' β' : ℝ), α' ∈ Set.Ioo 0 (Real.pi / 2) ∧
                 β' ∈ Set.Icc 0 (Real.pi / 2) ∧
                 x = 2 * α' - β' / 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_expression_l1720_172084


namespace NUMINAMATH_CALUDE_min_votes_class_president_l1720_172014

/-- Represents the minimum number of votes needed to win an election -/
def min_votes_to_win (total_votes : ℕ) (num_candidates : ℕ) : ℕ :=
  (total_votes / num_candidates) + 1

/-- Theorem: In an election with 4 candidates and 61 votes, the minimum number of votes to win is 16 -/
theorem min_votes_class_president : min_votes_to_win 61 4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_min_votes_class_president_l1720_172014


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1720_172032

def A : Set ℝ := {x | |x| ≤ 2}
def B : Set ℝ := {x | x^2 - 1 ≥ 0}

theorem intersection_of_A_and_B : 
  A ∩ B = {x : ℝ | -2 ≤ x ∧ x ≤ -1 ∨ 1 ≤ x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1720_172032


namespace NUMINAMATH_CALUDE_hidden_face_sum_l1720_172062

/-- Represents a standard six-sided die -/
def Die := Fin 6

/-- The sum of all numbers on a standard die -/
def dieTotalSum : ℕ := 21

/-- The visible numbers on the stacked dice -/
def visibleNumbers : List ℕ := [2, 2, 3, 3, 4, 5, 6]

/-- The number of dice stacked -/
def numberOfDice : ℕ := 3

theorem hidden_face_sum :
  (numberOfDice * dieTotalSum) - visibleNumbers.sum = 38 := by
  sorry

end NUMINAMATH_CALUDE_hidden_face_sum_l1720_172062


namespace NUMINAMATH_CALUDE_james_has_43_oreos_l1720_172033

/-- The number of Oreos James has -/
def james_oreos (jordan_oreos : ℕ) : ℕ := 4 * jordan_oreos + 7

/-- The total number of Oreos -/
def total_oreos : ℕ := 52

theorem james_has_43_oreos :
  ∃ (jordan_oreos : ℕ), 
    james_oreos jordan_oreos + jordan_oreos = total_oreos ∧
    james_oreos jordan_oreos = 43 := by
  sorry

end NUMINAMATH_CALUDE_james_has_43_oreos_l1720_172033


namespace NUMINAMATH_CALUDE_intersection_of_E_and_F_l1720_172091

def E : Set ℝ := {θ | Real.cos θ < Real.sin θ ∧ 0 ≤ θ ∧ θ ≤ 2 * Real.pi}
def F : Set ℝ := {θ | Real.tan θ < Real.sin θ}

theorem intersection_of_E_and_F :
  E ∩ F = {θ | Real.pi / 2 < θ ∧ θ < Real.pi} := by sorry

end NUMINAMATH_CALUDE_intersection_of_E_and_F_l1720_172091


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1720_172087

theorem sufficient_not_necessary : 
  (∀ x : ℝ, |x| < 2 → x^2 - x - 6 < 0) ∧ 
  (∃ x : ℝ, x^2 - x - 6 < 0 ∧ |x| ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1720_172087


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1720_172075

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 - 5*x + 15/2*a > 0) ↔ a > 5/6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1720_172075


namespace NUMINAMATH_CALUDE_max_product_value_l1720_172054

-- Define the functions f and h
def f : ℝ → ℝ := sorry
def h : ℝ → ℝ := sorry

-- State the theorem
theorem max_product_value :
  (∀ x, -3 ≤ f x ∧ f x ≤ 4) →
  (∀ x, -1 ≤ h x ∧ h x ≤ 3) →
  (∃ d, ∀ x, f x * h x ≤ d) ∧
  ∀ d', (∀ x, f x * h x ≤ d') → d' ≥ 12 :=
by sorry

end NUMINAMATH_CALUDE_max_product_value_l1720_172054


namespace NUMINAMATH_CALUDE_square_circle_area_ratio_l1720_172036

/-- Given a square and a circle intersecting such that each side of the square contains
    a chord of the circle with length equal to half the radius of the circle,
    the ratio of the area of the square to the area of the circle is 3/π. -/
theorem square_circle_area_ratio (r : ℝ) (h : r > 0) :
  let s := r * Real.sqrt 3
  (s^2) / (π * r^2) = 3 / π :=
by sorry

end NUMINAMATH_CALUDE_square_circle_area_ratio_l1720_172036


namespace NUMINAMATH_CALUDE_modified_mindmaster_codes_l1720_172097

/-- The number of possible secret codes in a modified Mindmaster game -/
def secret_codes (num_colors : ℕ) (num_slots : ℕ) : ℕ :=
  num_colors ^ num_slots

/-- Theorem: The number of secret codes in a game with 5 colors and 6 slots is 15625 -/
theorem modified_mindmaster_codes :
  secret_codes 5 6 = 15625 := by
  sorry

end NUMINAMATH_CALUDE_modified_mindmaster_codes_l1720_172097


namespace NUMINAMATH_CALUDE_problem_solution_l1720_172070

theorem problem_solution :
  (∃ m_max : ℝ, 
    (∀ m : ℝ, (∀ x : ℝ, |x + 3| + |x + m| ≥ 2 * m) → m ≤ m_max) ∧
    (∀ x : ℝ, |x + 3| + |x + m_max| ≥ 2 * m_max) ∧
    m_max = 1) ∧
  (∀ a b c : ℝ, 
    a > 0 → b > 0 → c > 0 → a + b + c = 1 →
    2 * a^2 + 3 * b^2 + 4 * c^2 ≥ 12/13 ∧
    (2 * a^2 + 3 * b^2 + 4 * c^2 = 12/13 ↔ a = 6/13 ∧ b = 4/13 ∧ c = 3/13)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1720_172070


namespace NUMINAMATH_CALUDE_base_k_representation_l1720_172024

/-- Represents a repeating decimal in base k -/
def repeating_decimal (a b : ℕ) (k : ℕ) : ℚ :=
  (a * k + b) / (k^2 - 1)

/-- The problem statement -/
theorem base_k_representation :
  ∃! k : ℕ, k > 0 ∧ 9 / 61 = repeating_decimal 3 4 k :=
sorry

end NUMINAMATH_CALUDE_base_k_representation_l1720_172024


namespace NUMINAMATH_CALUDE_game_probability_l1720_172026

theorem game_probability (n : ℕ) (p_alex p_mel p_chelsea : ℝ) : 
  n = 8 →
  p_alex = 1/2 →
  p_mel = 3 * p_chelsea →
  p_alex + p_mel + p_chelsea = 1 →
  (n.choose 4 * n.choose 3 * n.choose 1) * p_alex^4 * p_mel^3 * p_chelsea = 945/8192 :=
by sorry

end NUMINAMATH_CALUDE_game_probability_l1720_172026


namespace NUMINAMATH_CALUDE_octagon_interior_angles_sum_l1720_172042

/-- The sum of interior angles of a polygon with n sides is (n - 2) * 180 degrees -/
def sum_interior_angles (n : ℕ) : ℕ := (n - 2) * 180

/-- An octagon is a polygon with 8 sides -/
def octagon_sides : ℕ := 8

/-- The sum of the interior angles of an octagon is 1080 degrees -/
theorem octagon_interior_angles_sum :
  sum_interior_angles octagon_sides = 1080 := by
  sorry

end NUMINAMATH_CALUDE_octagon_interior_angles_sum_l1720_172042


namespace NUMINAMATH_CALUDE_arithmetic_sequence_unique_n_l1720_172015

/-- An arithmetic sequence with n terms, where a₁ is the first term and d is the common difference. -/
structure ArithmeticSequence where
  n : ℕ
  a₁ : ℚ
  d : ℚ

/-- The sum of the first k terms of an arithmetic sequence. -/
def sum_first_k (seq : ArithmeticSequence) (k : ℕ) : ℚ :=
  k * (2 * seq.a₁ + (k - 1) * seq.d) / 2

/-- The sum of the last k terms of an arithmetic sequence. -/
def sum_last_k (seq : ArithmeticSequence) (k : ℕ) : ℚ :=
  k * (2 * (seq.a₁ + (seq.n - k) * seq.d) + (k - 1) * seq.d) / 2

/-- The sum of all terms in an arithmetic sequence. -/
def sum_all (seq : ArithmeticSequence) : ℚ :=
  seq.n * (2 * seq.a₁ + (seq.n - 1) * seq.d) / 2

theorem arithmetic_sequence_unique_n (seq : ArithmeticSequence) :
  sum_first_k seq 4 = 40 →
  sum_last_k seq 4 = 80 →
  sum_all seq = 210 →
  seq.n = 14 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_unique_n_l1720_172015


namespace NUMINAMATH_CALUDE_pecan_pies_count_l1720_172055

/-- The number of pecan pies baked by Mrs. Hilt -/
def pecan_pies : ℝ := 16

/-- The number of apple pies baked by Mrs. Hilt -/
def apple_pies : ℝ := 14

/-- The factor by which the total number of pies needs to be increased -/
def increase_factor : ℝ := 5

/-- The total number of pies needed -/
def total_pies_needed : ℝ := 150

/-- Theorem stating that the number of pecan pies is correct given the conditions -/
theorem pecan_pies_count : 
  increase_factor * (pecan_pies + apple_pies) = total_pies_needed := by
  sorry

end NUMINAMATH_CALUDE_pecan_pies_count_l1720_172055


namespace NUMINAMATH_CALUDE_binomial_coefficient_relation_l1720_172068

theorem binomial_coefficient_relation (n : ℕ) : 
  (Nat.choose n 3 = 7 * Nat.choose n 1) → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_relation_l1720_172068


namespace NUMINAMATH_CALUDE_total_cost_of_kept_shirts_l1720_172057

def all_shirts : List ℕ := [15, 18, 20, 15, 25, 30, 20, 17, 22, 23, 29]
def returned_shirts : List ℕ := [20, 25, 30, 22, 23, 29]

theorem total_cost_of_kept_shirts :
  (all_shirts.sum - returned_shirts.sum) = 85 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_of_kept_shirts_l1720_172057


namespace NUMINAMATH_CALUDE_function_properties_l1720_172048

def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + (b - 8) * x - a - a * b

theorem function_properties (a b : ℝ) :
  (∀ x ∈ Set.Ioo (-3) 2, f a b x > 0) ∧
  (∀ x ∈ Set.Iic (-3) ∪ Set.Ici 2, f a b x < 0) →
  (∃ a₀ b₀ : ℝ, ∀ x, f a b x = -3 * x^2 - 3 * x + 18) ∧
  (∀ c : ℝ, (∀ x : ℝ, a * x^2 + b * x + c ≤ 0) ↔ c ≤ -25/12) ∧
  (∃ M : ℝ, M = -3 ∧ ∀ x > -1, (f a b x - 21) / (x + 1) ≤ M) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l1720_172048


namespace NUMINAMATH_CALUDE_gcd_720_90_minus_10_l1720_172080

theorem gcd_720_90_minus_10 : Nat.gcd 720 90 - 10 = 80 := by
  sorry

end NUMINAMATH_CALUDE_gcd_720_90_minus_10_l1720_172080


namespace NUMINAMATH_CALUDE_x_is_even_l1720_172005

theorem x_is_even (x : ℤ) (h : ∃ (k : ℤ), (2 * x) / 3 - x / 6 = k) : ∃ (m : ℤ), x = 2 * m := by
  sorry

end NUMINAMATH_CALUDE_x_is_even_l1720_172005


namespace NUMINAMATH_CALUDE_conic_is_hyperbola_l1720_172072

/-- The equation of the conic section -/
def conic_equation (x y : ℝ) : Prop :=
  (x + 5)^2 = (4*y - 3)^2 - 140

/-- Definition of a hyperbola -/
def is_hyperbola (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b c d e : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ a * b < 0 ∧
    ∀ x y, f x y ↔ a * x^2 + b * y^2 + c * x + d * y + e = 0

/-- Theorem stating that the given equation represents a hyperbola -/
theorem conic_is_hyperbola : is_hyperbola conic_equation :=
sorry

end NUMINAMATH_CALUDE_conic_is_hyperbola_l1720_172072


namespace NUMINAMATH_CALUDE_base_five_digits_of_3125_l1720_172076

theorem base_five_digits_of_3125 : ∃ n : ℕ, n = 6 ∧ 
  (∀ k : ℕ, 5^k ≤ 3125 → k + 1 ≤ n) ∧
  (∀ m : ℕ, (∀ k : ℕ, 5^k ≤ 3125 → k + 1 ≤ m) → n ≤ m) :=
by sorry

end NUMINAMATH_CALUDE_base_five_digits_of_3125_l1720_172076


namespace NUMINAMATH_CALUDE_sqrt_29_minus_1_between_4_and_5_l1720_172038

theorem sqrt_29_minus_1_between_4_and_5 :
  let a : ℝ := Real.sqrt 29 - 1
  4 < a ∧ a < 5 := by sorry

end NUMINAMATH_CALUDE_sqrt_29_minus_1_between_4_and_5_l1720_172038


namespace NUMINAMATH_CALUDE_chocolate_bar_pieces_l1720_172001

theorem chocolate_bar_pieces :
  ∀ (total : ℕ),
  (total / 2 : ℕ) + (total / 4 : ℕ) + 15 = total →
  total = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_chocolate_bar_pieces_l1720_172001


namespace NUMINAMATH_CALUDE_point_on_angle_bisector_l1720_172017

/-- 
Given a point M with coordinates (3n-2, 2n+7) that lies on the angle bisector 
of the second and fourth quadrants, prove that n = -1.
-/
theorem point_on_angle_bisector (n : ℝ) : 
  (∃ M : ℝ × ℝ, M.1 = 3*n - 2 ∧ M.2 = 2*n + 7 ∧ 
   M.1 + M.2 = 0) → n = -1 := by
sorry

end NUMINAMATH_CALUDE_point_on_angle_bisector_l1720_172017


namespace NUMINAMATH_CALUDE_perimeter_after_adding_tiles_l1720_172028

/-- Represents a configuration of square tiles -/
structure TileConfiguration where
  length : ℕ
  width : ℕ
  perimeter : ℕ

/-- The initial configuration of tiles -/
def initial_config : TileConfiguration :=
  { length := 6, width := 1, perimeter := 14 }

/-- Calculates the new perimeter after adding tiles -/
def new_perimeter (config : TileConfiguration) (added_tiles : ℕ) : ℕ :=
  2 * (config.length + added_tiles) + 2 * config.width

/-- Theorem stating that adding two tiles results in a perimeter of 18 -/
theorem perimeter_after_adding_tiles :
  new_perimeter initial_config 2 = 18 := by
  sorry

#eval new_perimeter initial_config 2

end NUMINAMATH_CALUDE_perimeter_after_adding_tiles_l1720_172028


namespace NUMINAMATH_CALUDE_sum_of_four_consecutive_squares_not_divisible_by_5_or_13_l1720_172047

theorem sum_of_four_consecutive_squares_not_divisible_by_5_or_13 (n : ℤ) :
  ∃ (k : ℤ), k ≠ 0 ∧ ((n - 1)^2 + n^2 + (n + 1)^2 + (n + 2)^2) % 5 = k ∧
              ((n - 1)^2 + n^2 + (n + 1)^2 + (n + 2)^2) % 13 = k :=
by sorry

end NUMINAMATH_CALUDE_sum_of_four_consecutive_squares_not_divisible_by_5_or_13_l1720_172047


namespace NUMINAMATH_CALUDE_alloy_composition_l1720_172089

theorem alloy_composition (gold_weight copper_weight alloy_weight : ℝ) 
  (h1 : gold_weight = 19)
  (h2 : alloy_weight = 17)
  (h3 : (4 * gold_weight + copper_weight) / 5 = alloy_weight) : 
  copper_weight = 9 := by
  sorry

end NUMINAMATH_CALUDE_alloy_composition_l1720_172089


namespace NUMINAMATH_CALUDE_income_expenditure_ratio_l1720_172029

/-- Given a person's income and savings, prove the ratio of income to expenditure -/
theorem income_expenditure_ratio 
  (income : ℕ) 
  (savings : ℕ) 
  (h1 : income = 36000) 
  (h2 : savings = 4000) :
  (income : ℚ) / (income - savings) = 9 / 8 := by
  sorry

end NUMINAMATH_CALUDE_income_expenditure_ratio_l1720_172029


namespace NUMINAMATH_CALUDE_x_value_at_stop_l1720_172053

/-- Represents the state of the computation at each step -/
structure State where
  x : ℕ
  s : ℕ

/-- Computes the next state given the current state -/
def nextState (state : State) : State :=
  { x := state.x + 3,
    s := state.s + state.x + 3 }

/-- Checks if the stopping condition is met -/
def isStoppingState (state : State) : Prop :=
  state.s ≥ 15000

/-- Represents the sequence of states -/
def stateSequence : ℕ → State
  | 0 => { x := 5, s := 0 }
  | n + 1 => nextState (stateSequence n)

theorem x_value_at_stop :
  ∃ n : ℕ, isStoppingState (stateSequence n) ∧
    ¬isStoppingState (stateSequence (n - 1)) ∧
    (stateSequence n).x = 368 :=
  sorry

end NUMINAMATH_CALUDE_x_value_at_stop_l1720_172053


namespace NUMINAMATH_CALUDE_horner_method_v3_l1720_172052

def f (x : ℝ) : ℝ := x^5 + 2*x^4 + x^3 - x^2 + 3*x - 5

def horner_v3 (a₅ a₄ a₃ a₂ a₁ a₀ x : ℝ) : ℝ :=
  ((((a₅ * x + a₄) * x + a₃) * x + a₂) * x + a₁) * x + a₀

theorem horner_method_v3 :
  horner_v3 1 2 1 (-1) 3 (-5) 5 = 179 :=
sorry

end NUMINAMATH_CALUDE_horner_method_v3_l1720_172052


namespace NUMINAMATH_CALUDE_objects_meet_distance_l1720_172006

/-- The distance traveled by object A when it meets object B -/
def distance_A_traveled (t : ℝ) : ℝ := t^2 - t

/-- The distance traveled by object B when it meets object A -/
def distance_B_traveled (t : ℝ) : ℝ := t + 4 * t^2

/-- The initial distance between objects A and B -/
def initial_distance : ℝ := 405

theorem objects_meet_distance (t : ℝ) (h : t > 0) 
  (h1 : distance_A_traveled t + distance_B_traveled t = initial_distance) : 
  distance_A_traveled t = 72 := by
  sorry

end NUMINAMATH_CALUDE_objects_meet_distance_l1720_172006


namespace NUMINAMATH_CALUDE_first_player_wins_l1720_172022

/-- Represents a digit (0-9) -/
def Digit : Type := Fin 10

/-- Represents an operation (addition or multiplication) -/
inductive Operation
| add : Operation
| mul : Operation

/-- Represents a game state -/
structure GameState :=
  (digits : List Digit)
  (operations : List Operation)

/-- Represents a game move -/
structure Move :=
  (digit : Digit)
  (operation : Option Operation)

/-- Evaluates the final result of the game -/
def evaluateGame (state : GameState) : ℕ :=
  sorry

/-- Checks if a number is even -/
def isEven (n : ℕ) : Prop :=
  ∃ k, n = 2 * k

/-- Theorem: The first player can always win with optimal play -/
theorem first_player_wins :
  ∀ (initial_digit : Digit),
    isEven initial_digit.val →
    ∃ (strategy : List Move),
      ∀ (opponent_moves : List Move),
        let final_state := sorry
        isEven (evaluateGame final_state) :=
by sorry

end NUMINAMATH_CALUDE_first_player_wins_l1720_172022


namespace NUMINAMATH_CALUDE_prob_one_black_in_three_draws_l1720_172079

-- Define the number of balls
def total_balls : ℕ := 6
def black_balls : ℕ := 2
def white_balls : ℕ := 4

-- Define the number of draws
def num_draws : ℕ := 3

-- Define the probability of drawing a black ball
def prob_black : ℚ := black_balls / total_balls

-- Define the probability of drawing a white ball
def prob_white : ℚ := white_balls / total_balls

-- Define the number of ways to choose 1 draw out of 3
def ways_to_choose : ℕ := 3

-- Theorem to prove
theorem prob_one_black_in_three_draws : 
  ways_to_choose * prob_black * prob_white^2 = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_prob_one_black_in_three_draws_l1720_172079


namespace NUMINAMATH_CALUDE_barley_percentage_is_80_percent_l1720_172037

/-- Represents the percentage of land that is cleared -/
def cleared_percentage : ℝ := 0.9

/-- Represents the percentage of cleared land planted with potato -/
def potato_percentage : ℝ := 0.1

/-- Represents the area of cleared land planted with tomato in acres -/
def tomato_area : ℝ := 90

/-- Represents the approximate total land area in acres -/
def total_land : ℝ := 1000

/-- Theorem stating that the percentage of cleared land planted with barley is 80% -/
theorem barley_percentage_is_80_percent :
  let cleared_land := cleared_percentage * total_land
  let barley_percentage := 1 - potato_percentage - (tomato_area / cleared_land)
  barley_percentage = 0.8 := by sorry

end NUMINAMATH_CALUDE_barley_percentage_is_80_percent_l1720_172037


namespace NUMINAMATH_CALUDE_powderman_distance_powderman_runs_185_yards_l1720_172044

/-- The distance in yards that a powderman runs when he hears a blast, given specific conditions -/
theorem powderman_distance (fuse_time reaction_time : ℝ) (run_speed : ℝ) (sound_speed : ℝ) : ℝ :=
  let blast_time := fuse_time
  let powderman_speed_ft_per_sec := run_speed * 3 -- Convert yards/sec to feet/sec
  let time_of_hearing := (sound_speed * blast_time + powderman_speed_ft_per_sec * reaction_time) / (sound_speed - powderman_speed_ft_per_sec)
  let distance_ft := powderman_speed_ft_per_sec * (time_of_hearing - reaction_time)
  let distance_yd := distance_ft / 3
  distance_yd

/-- The powderman runs 185 yards before hearing the blast under the given conditions -/
theorem powderman_runs_185_yards : 
  powderman_distance 20 2 10 1100 = 185 := by
  sorry


end NUMINAMATH_CALUDE_powderman_distance_powderman_runs_185_yards_l1720_172044


namespace NUMINAMATH_CALUDE_pure_imaginary_second_quadrant_l1720_172051

-- Define the complex number z as a function of real number m
def z (m : ℝ) : ℂ := Complex.mk (m^2 - 2*m - 3) (m^2 + 3*m + 2)

-- Theorem 1: z is a pure imaginary number if and only if m = 3
theorem pure_imaginary (m : ℝ) : z m = Complex.I * (z m).im ↔ m = 3 := by
  sorry

-- Theorem 2: z is in the second quadrant if and only if -1 < m < 3
theorem second_quadrant (m : ℝ) : (z m).re < 0 ∧ (z m).im > 0 ↔ -1 < m ∧ m < 3 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_second_quadrant_l1720_172051


namespace NUMINAMATH_CALUDE_range_of_m_l1720_172035

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, |x - 3| + |x - 5| < m^2 - m) → m < -1 ∨ m > 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l1720_172035


namespace NUMINAMATH_CALUDE_train_passing_time_symmetry_l1720_172096

theorem train_passing_time_symmetry 
  (fast_train_length slow_train_length : ℝ)
  (time_slow_passes_fast : ℝ)
  (fast_train_length_pos : 0 < fast_train_length)
  (slow_train_length_pos : 0 < slow_train_length)
  (time_slow_passes_fast_pos : 0 < time_slow_passes_fast) :
  let total_length := fast_train_length + slow_train_length
  let relative_speed := total_length / time_slow_passes_fast
  total_length / relative_speed = time_slow_passes_fast :=
by sorry

end NUMINAMATH_CALUDE_train_passing_time_symmetry_l1720_172096


namespace NUMINAMATH_CALUDE_max_value_of_f_l1720_172074

/-- The function f represents the quadratic equation y = -3x^2 + 12x + 4 -/
def f (x : ℝ) : ℝ := -3 * x^2 + 12 * x + 4

/-- The theorem states that the maximum value of f is 16 and occurs at x = 2 -/
theorem max_value_of_f :
  (∃ (x_max : ℝ), f x_max = 16 ∧ ∀ (x : ℝ), f x ≤ f x_max) ∧
  (f 2 = 16 ∧ ∀ (x : ℝ), f x ≤ 16) :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1720_172074


namespace NUMINAMATH_CALUDE_multiple_of_17_l1720_172008

theorem multiple_of_17 (x y : ℤ) : (2 * x + 3 * y) % 17 = 0 → (9 * x + 5 * y) % 17 = 0 := by
  sorry

end NUMINAMATH_CALUDE_multiple_of_17_l1720_172008


namespace NUMINAMATH_CALUDE_train_trip_time_difference_l1720_172065

/-- The time difference between two trips with given distance and speeds -/
theorem train_trip_time_difference 
  (distance : ℝ) 
  (speed_outbound speed_return : ℝ) 
  (h1 : distance = 480) 
  (h2 : speed_outbound = 160) 
  (h3 : speed_return = 120) : 
  distance / speed_return - distance / speed_outbound = 1 := by
sorry

end NUMINAMATH_CALUDE_train_trip_time_difference_l1720_172065


namespace NUMINAMATH_CALUDE_cost_mms_in_snickers_l1720_172058

/-- The cost of a pack of M&M's in terms of Snickers pieces -/
theorem cost_mms_in_snickers 
  (snickers_quantity : ℕ)
  (mms_quantity : ℕ)
  (snickers_price : ℚ)
  (total_paid : ℚ)
  (change_received : ℚ)
  (h1 : snickers_quantity = 2)
  (h2 : mms_quantity = 3)
  (h3 : snickers_price = 3/2)
  (h4 : total_paid = 20)
  (h5 : change_received = 8) :
  (total_paid - change_received - snickers_quantity * snickers_price) / mms_quantity = 2 * snickers_price :=
by sorry

end NUMINAMATH_CALUDE_cost_mms_in_snickers_l1720_172058


namespace NUMINAMATH_CALUDE_union_equals_universal_l1720_172067

def U : Set ℕ := {2, 3, 4, 5, 6, 7}
def M : Set ℕ := {3, 4, 5, 7}
def N : Set ℕ := {2, 4, 5, 6}

theorem union_equals_universal : M ∪ N = U := by
  sorry

end NUMINAMATH_CALUDE_union_equals_universal_l1720_172067


namespace NUMINAMATH_CALUDE_marble_distribution_l1720_172018

theorem marble_distribution (total_marbles : ℕ) (num_groups : ℕ) (marbles_per_group : ℕ) :
  total_marbles = 64 →
  num_groups = 32 →
  total_marbles = num_groups * marbles_per_group →
  marbles_per_group = 2 := by
  sorry

end NUMINAMATH_CALUDE_marble_distribution_l1720_172018


namespace NUMINAMATH_CALUDE_fraction_value_l1720_172095

theorem fraction_value (x y : ℝ) (h1 : 1 < (x - y) / (x + y)) (h2 : (x - y) / (x + y) < 3) 
  (h3 : ∃ (n : ℤ), x / y = n) : x / y = -2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l1720_172095


namespace NUMINAMATH_CALUDE_water_left_proof_l1720_172019

def water_problem (initial_amount mother_drink father_extra sister_drink : ℝ) : Prop :=
  let father_drink := mother_drink + father_extra
  let total_consumed := mother_drink + father_drink + sister_drink
  let water_left := initial_amount - total_consumed
  water_left = 0.3

theorem water_left_proof :
  water_problem 1 0.1 0.2 0.3 := by
  sorry

end NUMINAMATH_CALUDE_water_left_proof_l1720_172019


namespace NUMINAMATH_CALUDE_gcd_of_72_120_168_l1720_172090

theorem gcd_of_72_120_168 : Nat.gcd 72 (Nat.gcd 120 168) = 24 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_72_120_168_l1720_172090


namespace NUMINAMATH_CALUDE_tan_420_degrees_l1720_172009

theorem tan_420_degrees : Real.tan (420 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_420_degrees_l1720_172009


namespace NUMINAMATH_CALUDE_minimum_time_two_people_one_bicycle_l1720_172003

/-- The minimum time problem for two people traveling with one bicycle -/
theorem minimum_time_two_people_one_bicycle
  (distance : ℝ)
  (walk_speed1 walk_speed2 bike_speed1 bike_speed2 : ℝ)
  (h_distance : distance = 40)
  (h_walk_speed1 : walk_speed1 = 4)
  (h_walk_speed2 : walk_speed2 = 6)
  (h_bike_speed1 : bike_speed1 = 30)
  (h_bike_speed2 : bike_speed2 = 20)
  (h_positive : walk_speed1 > 0 ∧ walk_speed2 > 0 ∧ bike_speed1 > 0 ∧ bike_speed2 > 0) :
  ∃ (t : ℝ), t = 25/9 ∧ 
  ∀ (t' : ℝ), (∃ (x y : ℝ), 
    x ≥ 0 ∧ y ≥ 0 ∧
    bike_speed1 * x + walk_speed1 * y = distance ∧
    walk_speed2 * x + bike_speed2 * y = distance ∧
    t' = x + y) → t ≤ t' :=
by sorry

end NUMINAMATH_CALUDE_minimum_time_two_people_one_bicycle_l1720_172003


namespace NUMINAMATH_CALUDE_village_population_l1720_172021

theorem village_population (initial_population : ℕ) 
  (death_rate : ℚ) (leaving_rate : ℚ) : 
  initial_population = 4500 →
  death_rate = 1/10 →
  leaving_rate = 1/5 →
  (initial_population - initial_population * death_rate) * (1 - leaving_rate) = 3240 :=
by sorry

end NUMINAMATH_CALUDE_village_population_l1720_172021


namespace NUMINAMATH_CALUDE_complex_fraction_calculation_l1720_172027

theorem complex_fraction_calculation : (9 * 9 - 2 * 2) / ((1 / 12) - (1 / 19)) = 2508 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_calculation_l1720_172027


namespace NUMINAMATH_CALUDE_power_of_256_l1720_172098

theorem power_of_256 : (256 : ℝ) ^ (4/5 : ℝ) = 64 := by
  have h1 : 256 = 2^8 := by sorry
  sorry

end NUMINAMATH_CALUDE_power_of_256_l1720_172098


namespace NUMINAMATH_CALUDE_harrys_pizza_toppings_l1720_172046

/-- Calculates the number of toppings per pizza given the conditions of Harry's pizza order --/
theorem harrys_pizza_toppings : ∀ (toppings_per_pizza : ℕ),
  (14 : ℚ) * 2 + -- Cost of two large pizzas
  (2 : ℚ) * (2 * toppings_per_pizza) + -- Cost of toppings
  (((14 : ℚ) * 2 + (2 : ℚ) * (2 * toppings_per_pizza)) * (1 / 4)) -- 25% tip
  = 50 →
  toppings_per_pizza = 3 := by
  sorry

#check harrys_pizza_toppings

end NUMINAMATH_CALUDE_harrys_pizza_toppings_l1720_172046


namespace NUMINAMATH_CALUDE_rational_numbers_composition_l1720_172066

-- Define the set of integers
def Integers : Set ℚ := {x : ℚ | ∃ n : ℤ, x = n}

-- Define the set of fractions
def Fractions : Set ℚ := {x : ℚ | ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b}

-- Theorem statement
theorem rational_numbers_composition :
  Set.univ = Integers ∪ Fractions :=
sorry

end NUMINAMATH_CALUDE_rational_numbers_composition_l1720_172066


namespace NUMINAMATH_CALUDE_chord_length_of_intersecting_curves_l1720_172049

/-- The length of the chord formed by the intersection of two curves in polar coordinates -/
theorem chord_length_of_intersecting_curves (C₁ C₂ : ℝ → ℝ → Prop) :
  (∀ ρ θ, C₁ ρ θ ↔ ρ = 2 * Real.sin θ) →
  (∀ ρ θ, C₂ ρ θ ↔ θ = Real.pi / 3) →
  ∃ M N : ℝ × ℝ,
    (∃ ρ₁ θ₁, C₁ ρ₁ θ₁ ∧ M = (ρ₁ * Real.cos θ₁, ρ₁ * Real.sin θ₁)) ∧
    (∃ ρ₂ θ₂, C₂ ρ₂ θ₂ ∧ M = (ρ₂ * Real.cos θ₂, ρ₂ * Real.sin θ₂)) ∧
    (∃ ρ₃ θ₃, C₁ ρ₃ θ₃ ∧ N = (ρ₃ * Real.cos θ₃, ρ₃ * Real.sin θ₃)) ∧
    (∃ ρ₄ θ₄, C₂ ρ₄ θ₄ ∧ N = (ρ₄ * Real.cos θ₄, ρ₄ * Real.sin θ₄)) ∧
    Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_chord_length_of_intersecting_curves_l1720_172049


namespace NUMINAMATH_CALUDE_milk_cost_verify_milk_cost_l1720_172034

/-- Proves that the cost of a gallon of milk is $4 given the conditions about coffee consumption and costs --/
theorem milk_cost (cups_per_day : ℕ) (oz_per_cup : ℚ) (bag_cost : ℚ) (oz_per_bag : ℚ) 
  (milk_usage : ℚ) (total_cost : ℚ) : ℚ :=
by
  -- Define the conditions
  have h1 : cups_per_day = 2 := by sorry
  have h2 : oz_per_cup = 3/2 := by sorry
  have h3 : bag_cost = 8 := by sorry
  have h4 : oz_per_bag = 21/2 := by sorry
  have h5 : milk_usage = 1/2 := by sorry
  have h6 : total_cost = 18 := by sorry

  -- Calculate the cost of a gallon of milk
  sorry

/-- The cost of a gallon of milk --/
def gallon_milk_cost : ℚ := 4

/-- Proves that the calculated cost matches the expected cost --/
theorem verify_milk_cost : 
  milk_cost 2 (3/2) 8 (21/2) (1/2) 18 = gallon_milk_cost := by sorry

end NUMINAMATH_CALUDE_milk_cost_verify_milk_cost_l1720_172034


namespace NUMINAMATH_CALUDE_solution_equation1_solution_equation2_l1720_172013

-- Define the equations
def equation1 (x : ℝ) : Prop := 3 * x + 3 = 7 - x
def equation2 (x : ℝ) : Prop := (1/2) * x - 6 = (3/4) * x

-- Theorem for equation 1
theorem solution_equation1 : ∃! x : ℝ, equation1 x ∧ x = 1 := by sorry

-- Theorem for equation 2
theorem solution_equation2 : ∃! x : ℝ, equation2 x ∧ x = -24 := by sorry

end NUMINAMATH_CALUDE_solution_equation1_solution_equation2_l1720_172013


namespace NUMINAMATH_CALUDE_time_to_fill_leaking_pool_l1720_172063

/-- Time to fill a leaking pool -/
theorem time_to_fill_leaking_pool 
  (pool_capacity : ℝ) 
  (filling_rate : ℝ) 
  (leaking_rate : ℝ) 
  (h1 : pool_capacity = 60) 
  (h2 : filling_rate = 1.6) 
  (h3 : leaking_rate = 0.1) : 
  pool_capacity / (filling_rate - leaking_rate) = 40 := by
sorry

end NUMINAMATH_CALUDE_time_to_fill_leaking_pool_l1720_172063


namespace NUMINAMATH_CALUDE_angelina_speed_to_library_l1720_172078

-- Define the distances
def home_to_grocery : ℝ := 150
def grocery_to_gym : ℝ := 200
def gym_to_park : ℝ := 250
def park_to_library : ℝ := 300

-- Define Angelina's initial speed
def v : ℝ := 5

-- Define the theorem
theorem angelina_speed_to_library :
  let time_home_to_grocery := home_to_grocery / v
  let time_grocery_to_gym := grocery_to_gym / (2 * v)
  let time_gym_to_park := gym_to_park / (v / 2)
  let time_park_to_library := park_to_library / (6 * v)
  time_grocery_to_gym = time_home_to_grocery - 10 ∧
  time_gym_to_park = time_park_to_library + 20 →
  6 * v = 30 := by
  sorry

end NUMINAMATH_CALUDE_angelina_speed_to_library_l1720_172078


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l1720_172043

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_sum : a 2 + a 4 = 16)
  (h_first : a 1 = 1) :
  a 5 = 15 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l1720_172043


namespace NUMINAMATH_CALUDE_sqrt_sum_rational_implies_components_rational_l1720_172081

theorem sqrt_sum_rational_implies_components_rational
  (m n p : ℚ)
  (h : ∃ (q : ℚ), Real.sqrt m + Real.sqrt n + Real.sqrt p = q) :
  (∃ (r : ℚ), Real.sqrt m = r) ∧
  (∃ (s : ℚ), Real.sqrt n = s) ∧
  (∃ (t : ℚ), Real.sqrt p = t) :=
sorry

end NUMINAMATH_CALUDE_sqrt_sum_rational_implies_components_rational_l1720_172081


namespace NUMINAMATH_CALUDE_range_of_a_l1720_172025

/-- The function f(x) = a - x² -/
def f (a : ℝ) (x : ℝ) : ℝ := a - x^2

/-- The function g(x) = x + 2 -/
def g (x : ℝ) : ℝ := x + 2

/-- The theorem stating the range of a -/
theorem range_of_a (a : ℝ) : 
  (∃ x y : ℝ, 1 ≤ x ∧ x ≤ 2 ∧ f a x = -g y) → 
  -2 ≤ a ∧ a ≤ 0 := by
  sorry

#check range_of_a

end NUMINAMATH_CALUDE_range_of_a_l1720_172025


namespace NUMINAMATH_CALUDE_arctan_sum_greater_than_pi_half_l1720_172056

theorem arctan_sum_greater_than_pi_half (a b : ℝ) : 
  a = 2/3 → (a + 1) * (b + 1) = 3 → Real.arctan a + Real.arctan b > π/2 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_greater_than_pi_half_l1720_172056


namespace NUMINAMATH_CALUDE_notebook_savings_proof_l1720_172050

/-- Calculates the savings on a notebook purchase with discounts -/
def calculateSavings (originalPrice : ℝ) (quantity : ℕ) (saleDiscount : ℝ) (volumeDiscount : ℝ) : ℝ :=
  let discountedPrice := originalPrice * (1 - saleDiscount)
  let finalPrice := discountedPrice * (1 - volumeDiscount)
  quantity * (originalPrice - finalPrice)

/-- Proves that the savings on the notebook purchase is $7.84 -/
theorem notebook_savings_proof :
  calculateSavings 3 8 0.25 0.1 = 7.84 := by
  sorry

#eval calculateSavings 3 8 0.25 0.1

end NUMINAMATH_CALUDE_notebook_savings_proof_l1720_172050


namespace NUMINAMATH_CALUDE_a_2016_value_l1720_172012

def sequence_sum (n : ℕ) : ℕ := n ^ 2

theorem a_2016_value :
  let a : ℕ → ℕ := fun n => sequence_sum n - sequence_sum (n - 1)
  a 2016 = 4031 := by
  sorry

end NUMINAMATH_CALUDE_a_2016_value_l1720_172012


namespace NUMINAMATH_CALUDE_system_solution_l1720_172092

theorem system_solution :
  ∃ (x y : ℚ), 4 * x - 3 * y = 2 ∧ 5 * x + y = (3 / 2) ∧ x = (13 / 38) ∧ y = (-4 / 19) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1720_172092


namespace NUMINAMATH_CALUDE_perseverance_arrangements_l1720_172061

def word : String := "PERSEVERANCE"

def letter_count : Nat := word.length

def e_count : Nat := 4
def r_count : Nat := 2
def single_count : Nat := 6  -- P, S, V, A, N, C appear once each

theorem perseverance_arrangements :
  (letter_count.factorial / (e_count.factorial * r_count.factorial)) = 9979200 := by
  sorry

end NUMINAMATH_CALUDE_perseverance_arrangements_l1720_172061


namespace NUMINAMATH_CALUDE_john_photos_count_l1720_172011

/-- The number of photos each person brings and the total slots in the album --/
def photo_problem (cristina_photos sarah_photos clarissa_photos total_slots : ℕ) : Prop :=
  ∃ john_photos : ℕ,
    john_photos = total_slots - (cristina_photos + sarah_photos + clarissa_photos)

/-- Theorem stating that John brings 10 photos given the problem conditions --/
theorem john_photos_count :
  photo_problem 7 9 14 40 → ∃ john_photos : ℕ, john_photos = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_john_photos_count_l1720_172011


namespace NUMINAMATH_CALUDE_minor_arc_intercept_l1720_172082

/-- Given a circle x^2 + y^2 = 4 and a line y = -√3x + b, if the minor arc intercepted
    by the line on the circle corresponds to a central angle of 120°, then b = ±2 -/
theorem minor_arc_intercept (b : ℝ) : 
  (∀ x y, x^2 + y^2 = 4 → y = -Real.sqrt 3 * x + b) →
  (∃ θ, θ = 2 * Real.pi / 3) →
  (b = 2 ∨ b = -2) := by
  sorry

end NUMINAMATH_CALUDE_minor_arc_intercept_l1720_172082


namespace NUMINAMATH_CALUDE_starters_count_l1720_172088

/-- Represents a set of twins -/
structure TwinSet :=
  (twin1 : ℕ)
  (twin2 : ℕ)

/-- Represents a basketball team -/
structure BasketballTeam :=
  (total_players : ℕ)
  (twin_set1 : TwinSet)
  (twin_set2 : TwinSet)

/-- Calculates the number of ways to choose starters with twin restrictions -/
def choose_starters (team : BasketballTeam) (num_starters : ℕ) : ℕ :=
  sorry

/-- The specific basketball team in the problem -/
def problem_team : BasketballTeam :=
  { total_players := 18
  , twin_set1 := { twin1 := 1, twin2 := 2 }  -- Representing Ben & Jerry
  , twin_set2 := { twin1 := 3, twin2 := 4 }  -- Representing Tom & Tim
  }

theorem starters_count : choose_starters problem_team 5 = 1834 := by
  sorry

end NUMINAMATH_CALUDE_starters_count_l1720_172088


namespace NUMINAMATH_CALUDE_min_value_fraction_l1720_172059

theorem min_value_fraction (x : ℝ) (h : x > 2) : (x^2 - 4*x + 5) / (x - 2) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_fraction_l1720_172059


namespace NUMINAMATH_CALUDE_books_after_donation_l1720_172031

theorem books_after_donation (boris_initial : ℕ) (cameron_initial : ℕ) 
  (boris_donation_fraction : ℚ) (cameron_donation_fraction : ℚ)
  (h1 : boris_initial = 24)
  (h2 : cameron_initial = 30)
  (h3 : boris_donation_fraction = 1/4)
  (h4 : cameron_donation_fraction = 1/3) :
  (boris_initial - boris_initial * boris_donation_fraction).floor +
  (cameron_initial - cameron_initial * cameron_donation_fraction).floor = 38 := by
sorry

end NUMINAMATH_CALUDE_books_after_donation_l1720_172031


namespace NUMINAMATH_CALUDE_parabola_triangle_property_l1720_172000

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x

-- Define the line y = x + 3
def line (x y : ℝ) : Prop := y = x + 3

-- Define a point on the parabola
structure PointOnParabola (p : ℝ) where
  x : ℝ
  y : ℝ
  on_parabola : parabola p x y

-- Define the theorem
theorem parabola_triangle_property (p : ℝ) :
  parabola p 1 2 →  -- The parabola passes through (1, 2)
  ∀ (A : PointOnParabola p),
    A.x ≠ 1 ∨ A.y ≠ 2 →  -- A is different from (1, 2)
    ∃ (P B : ℝ × ℝ),
      -- P is on the line AC and y = x + 3
      (∃ t : ℝ, P.1 = (1 - t) * 1 + t * A.x ∧ P.2 = (1 - t) * 2 + t * A.y) ∧
      line P.1 P.2 ∧
      -- B is on the parabola and has the same y-coordinate as P
      parabola p B.1 B.2 ∧ B.2 = P.2 →
      -- 1. AB passes through (3, 2)
      (∃ s : ℝ, 3 = (1 - s) * A.x + s * B.1 ∧ 2 = (1 - s) * A.y + s * B.2) ∧
      -- 2. The minimum area of triangle ABC is 4√2
      (∀ (area : ℝ), area ≥ 0 ∧ area * area = 32 → 
        ∃ (A' : PointOnParabola p) (P' B' : ℝ × ℝ),
          A'.x ≠ 1 ∨ A'.y ≠ 2 ∧
          (∃ t : ℝ, P'.1 = (1 - t) * 1 + t * A'.x ∧ P'.2 = (1 - t) * 2 + t * A'.y) ∧
          line P'.1 P'.2 ∧
          parabola p B'.1 B'.2 ∧ B'.2 = P'.2 ∧
          area = (1/2) * Real.sqrt ((A'.x - 1)^2 + (A'.y - 2)^2) * Real.sqrt ((B'.1 - 1)^2 + (B'.2 - 2)^2)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_triangle_property_l1720_172000


namespace NUMINAMATH_CALUDE_part_one_part_two_l1720_172041

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x - 3|

-- Part 1
theorem part_one (a : ℝ) (h1 : a < 3) 
  (h2 : ∀ x, f a x ≥ 4 ↔ x ≤ 1/2 ∨ x ≥ 9/2) : 
  a = 2 := by sorry

-- Part 2
theorem part_two (a : ℝ) (h1 : a < 3) 
  (h2 : ∀ x, f a x + |x - 3| ≥ 1) : 
  a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1720_172041


namespace NUMINAMATH_CALUDE_sqrt_nine_subtraction_l1720_172077

theorem sqrt_nine_subtraction : 1 - Real.sqrt 9 = -2 := by sorry

end NUMINAMATH_CALUDE_sqrt_nine_subtraction_l1720_172077


namespace NUMINAMATH_CALUDE_seventh_group_sample_l1720_172004

/-- Represents the systematic sampling method described in the problem -/
def systematicSample (populationSize : ℕ) (groupCount : ℕ) (firstNumber : ℕ) (groupNumber : ℕ) : ℕ :=
  let interval := populationSize / groupCount
  let lastTwoDigits := (firstNumber + 33 * groupNumber) % 100
  (groupNumber - 1) * interval + lastTwoDigits

/-- Theorem stating the result of the systematic sampling for the 7th group -/
theorem seventh_group_sample :
  systematicSample 1000 10 57 7 = 688 := by
  sorry

#eval systematicSample 1000 10 57 7

end NUMINAMATH_CALUDE_seventh_group_sample_l1720_172004


namespace NUMINAMATH_CALUDE_chip_cost_is_correct_l1720_172094

/-- The cost of a bag of chips, given Amber's spending scenario -/
def chip_cost (total_money : ℚ) (candy_cost : ℚ) (candy_ounces : ℚ) (chip_ounces : ℚ) (max_ounces : ℚ) : ℚ :=
  total_money / (max_ounces / chip_ounces)

/-- Theorem stating that the cost of a bag of chips is $1.40 in Amber's scenario -/
theorem chip_cost_is_correct :
  chip_cost 7 1 12 17 85 = (14 : ℚ) / 10 := by
  sorry

end NUMINAMATH_CALUDE_chip_cost_is_correct_l1720_172094


namespace NUMINAMATH_CALUDE_tangent_length_from_point_to_circle_l1720_172064

/-- The length of the tangent from a point to a circle -/
theorem tangent_length_from_point_to_circle 
  (P : ℝ × ℝ) -- Point P
  (center : ℝ × ℝ) -- Center of the circle
  (r : ℝ) -- Radius of the circle
  (h1 : P = (2, 3)) -- P coordinates
  (h2 : center = (0, 0)) -- Circle center
  (h3 : r = 1) -- Circle radius
  (h4 : (P.1 - center.1)^2 + (P.2 - center.2)^2 > r^2) -- P is outside the circle
  : Real.sqrt ((P.1 - center.1)^2 + (P.2 - center.2)^2 - r^2) = 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_tangent_length_from_point_to_circle_l1720_172064


namespace NUMINAMATH_CALUDE_range_of_m_l1720_172086

-- Define proposition p
def p (m : ℝ) : Prop := ∀ x y : ℝ, x^2/2 + y^2/(m-1) = 1 → (m - 1 > 2)

-- Define proposition q
def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 - 2*x + 4*m ≠ 0

-- Define the theorem
theorem range_of_m (m : ℝ) : 
  (¬p m) ∧ (p m ∨ q m) → m ∈ Set.Ioo (1/4 : ℝ) 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1720_172086


namespace NUMINAMATH_CALUDE_eight_cubic_polynomials_l1720_172030

/-- A polynomial function of degree at most 3 -/
def CubicPolynomial (a b c d : ℝ) : ℝ → ℝ := fun x ↦ a * x^3 + b * x^2 + c * x + d

/-- The condition that f(x) f(-x) = f(x^3) for all x -/
def SatisfiesCondition (f : ℝ → ℝ) : Prop :=
  ∀ x, f x * f (-x) = f (x^3)

/-- The main theorem stating that there are exactly 8 cubic polynomials satisfying the condition -/
theorem eight_cubic_polynomials :
  ∃! (s : Finset (ℝ × ℝ × ℝ × ℝ)), 
    (∀ (a b c d : ℝ), (a, b, c, d) ∈ s ↔ SatisfiesCondition (CubicPolynomial a b c d)) ∧
    Finset.card s = 8 := by
  sorry


end NUMINAMATH_CALUDE_eight_cubic_polynomials_l1720_172030


namespace NUMINAMATH_CALUDE_circle_op_inequality_l1720_172039

def circle_op (x y : ℝ) : ℝ := x * (1 - y)

theorem circle_op_inequality (a : ℝ) : 
  (∀ x : ℝ, circle_op (x - a) (x + a) < 1) → -1/2 < a ∧ a < 3/2 := by
  sorry

end NUMINAMATH_CALUDE_circle_op_inequality_l1720_172039
