import Mathlib

namespace NUMINAMATH_CALUDE_inscribed_rectangle_sides_l538_53800

theorem inscribed_rectangle_sides (R : ℝ) (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (a^2 + b^2 = 4 * R^2) ∧ (a * b = (1/2) * π * R^2) →
  ((a = (R * Real.sqrt (π + 4) + R * Real.sqrt (4 - π)) / 2 ∧
    b = (R * Real.sqrt (π + 4) - R * Real.sqrt (4 - π)) / 2) ∨
   (a = (R * Real.sqrt (π + 4) - R * Real.sqrt (4 - π)) / 2 ∧
    b = (R * Real.sqrt (π + 4) + R * Real.sqrt (4 - π)) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_sides_l538_53800


namespace NUMINAMATH_CALUDE_blanket_price_problem_l538_53878

/-- Proves that the unknown rate of two blankets is 228.75, given the conditions of the problem -/
theorem blanket_price_problem (price_3 : ℕ) (price_1 : ℕ) (discount : ℚ) (tax : ℚ) (avg_price : ℕ) :
  price_3 = 100 →
  price_1 = 150 →
  discount = 1/10 →
  tax = 3/20 →
  avg_price = 150 →
  let total_blankets : ℕ := 6
  let discounted_price_3 : ℚ := 3 * price_3 * (1 - discount)
  let taxed_price_1 : ℚ := price_1 * (1 + tax)
  let total_price : ℚ := total_blankets * avg_price
  ∃ x : ℚ, 
    x = (total_price - discounted_price_3 - taxed_price_1) / 2 ∧ 
    x = 457.5 / 2 := by
  sorry

#eval 457.5 / 2  -- Should output 228.75

end NUMINAMATH_CALUDE_blanket_price_problem_l538_53878


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l538_53868

theorem geometric_sequence_sum (n : ℕ) : 
  let a := (1 : ℚ) / 3  -- first term
  let r := (1 : ℚ) / 2  -- common ratio
  let sum := a * (1 - r^n) / (1 - r)  -- sum formula for geometric sequence
  sum = 80 / 243 → n = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l538_53868


namespace NUMINAMATH_CALUDE_rancher_loss_calculation_l538_53805

/-- Represents the rancher's cattle situation and calculates the loss --/
def rancher_loss (initial_cattle : ℕ) (initial_total_price : ℕ) (dead_cattle : ℕ) (price_reduction : ℕ) : ℕ :=
  let original_price_per_head := initial_total_price / initial_cattle
  let new_price_per_head := original_price_per_head - price_reduction
  let remaining_cattle := initial_cattle - dead_cattle
  let new_total_price := new_price_per_head * remaining_cattle
  initial_total_price - new_total_price

/-- Theorem stating the rancher's loss given the problem conditions --/
theorem rancher_loss_calculation :
  rancher_loss 340 204000 172 150 = 128400 := by
  sorry

end NUMINAMATH_CALUDE_rancher_loss_calculation_l538_53805


namespace NUMINAMATH_CALUDE_chromium_percentage_in_combined_alloy_l538_53824

/-- Calculates the percentage of chromium in a new alloy formed by combining two other alloys -/
theorem chromium_percentage_in_combined_alloy 
  (chromium_percent1 : ℝ) 
  (weight1 : ℝ) 
  (chromium_percent2 : ℝ) 
  (weight2 : ℝ) 
  (h1 : chromium_percent1 = 12)
  (h2 : weight1 = 15)
  (h3 : chromium_percent2 = 8)
  (h4 : weight2 = 40) :
  let total_chromium := (chromium_percent1 / 100) * weight1 + (chromium_percent2 / 100) * weight2
  let total_weight := weight1 + weight2
  (total_chromium / total_weight) * 100 = 9.09 := by
  sorry

end NUMINAMATH_CALUDE_chromium_percentage_in_combined_alloy_l538_53824


namespace NUMINAMATH_CALUDE_laura_pants_purchase_l538_53865

def pants_cost : ℕ := 54
def shirt_cost : ℕ := 33
def num_shirts : ℕ := 4
def money_given : ℕ := 250
def change_received : ℕ := 10

theorem laura_pants_purchase :
  (money_given - change_received - num_shirts * shirt_cost) / pants_cost = 2 := by
  sorry

end NUMINAMATH_CALUDE_laura_pants_purchase_l538_53865


namespace NUMINAMATH_CALUDE_armands_guessing_game_l538_53829

theorem armands_guessing_game (x : ℤ) : x = 33 ↔ 3 * x = 2 * 51 - 3 := by
  sorry

end NUMINAMATH_CALUDE_armands_guessing_game_l538_53829


namespace NUMINAMATH_CALUDE_middle_pile_has_five_cards_l538_53836

/-- Represents the number of cards in each pile -/
structure CardPiles :=
  (left : ℕ)
  (middle : ℕ)
  (right : ℕ)

/-- The initial state of the card piles -/
def initial_state (x : ℕ) : CardPiles :=
  { left := x, middle := x, right := x }

/-- Condition that each pile has at least 2 cards initially -/
def valid_initial_state (s : CardPiles) : Prop :=
  s.left ≥ 2 ∧ s.middle ≥ 2 ∧ s.right ≥ 2

/-- The state after performing the four steps -/
def final_state (s : CardPiles) : CardPiles :=
  let step1 := s
  let step2 := { step1 with left := step1.left - 2, middle := step1.middle + 2 }
  let step3 := { step2 with right := step2.right - 1, middle := step2.middle + 1 }
  let step4 := { step3 with left := step3.left + step3.left, middle := step3.middle - step3.left }
  step4

/-- The main theorem stating that the middle pile always has 5 cards after the steps -/
theorem middle_pile_has_five_cards (x : ℕ) :
  let initial := initial_state x
  valid_initial_state initial →
  (final_state initial).middle = 5 :=
by sorry

end NUMINAMATH_CALUDE_middle_pile_has_five_cards_l538_53836


namespace NUMINAMATH_CALUDE_power_mod_prime_remainder_5_1000_mod_29_l538_53821

theorem power_mod_prime (p : Nat) (a : Nat) (m : Nat) (h : Prime p) :
  a ^ m % p = (a ^ (m % (p - 1))) % p :=
sorry

theorem remainder_5_1000_mod_29 : 5^1000 % 29 = 21 := by
  have h1 : Prime 29 := by sorry
  have h2 : 5^1000 % 29 = (5^(1000 % 28)) % 29 := by
    apply power_mod_prime 29 5 1000 h1
  have h3 : 1000 % 28 = 20 := by sorry
  have h4 : (5^20) % 29 = 21 := by sorry
  rw [h2, h3, h4]

end NUMINAMATH_CALUDE_power_mod_prime_remainder_5_1000_mod_29_l538_53821


namespace NUMINAMATH_CALUDE_katherines_fruits_l538_53809

theorem katherines_fruits (apples pears bananas : ℕ) : 
  apples = 4 →
  pears = 3 * apples →
  apples + pears + bananas = 21 →
  bananas = 5 := by
sorry

end NUMINAMATH_CALUDE_katherines_fruits_l538_53809


namespace NUMINAMATH_CALUDE_complex_product_magnitude_l538_53886

theorem complex_product_magnitude : Complex.abs ((20 - 15 * Complex.I) * (12 + 25 * Complex.I)) = 25 * Real.sqrt 769 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_magnitude_l538_53886


namespace NUMINAMATH_CALUDE_tetrahedron_altitude_exsphere_relation_l538_53841

/-- A tetrahedron with its altitudes and exsphere radii -/
structure Tetrahedron where
  h₁ : ℝ
  h₂ : ℝ
  h₃ : ℝ
  h₄ : ℝ
  r₁ : ℝ
  r₂ : ℝ
  r₃ : ℝ
  r₄ : ℝ
  h_pos : h₁ > 0 ∧ h₂ > 0 ∧ h₃ > 0 ∧ h₄ > 0
  r_pos : r₁ > 0 ∧ r₂ > 0 ∧ r₃ > 0 ∧ r₄ > 0

/-- The theorem about the relationship between altitudes and exsphere radii in a tetrahedron -/
theorem tetrahedron_altitude_exsphere_relation (t : Tetrahedron) :
  2 * (1 / t.h₁ + 1 / t.h₂ + 1 / t.h₃ + 1 / t.h₄) =
  1 / t.r₁ + 1 / t.r₂ + 1 / t.r₃ + 1 / t.r₄ := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_altitude_exsphere_relation_l538_53841


namespace NUMINAMATH_CALUDE_remainder_2027_div_28_l538_53858

theorem remainder_2027_div_28 : 2027 % 28 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2027_div_28_l538_53858


namespace NUMINAMATH_CALUDE_max_odd_digits_in_sum_l538_53899

/-- A function that counts the number of odd digits in a natural number -/
def count_odd_digits (n : ℕ) : ℕ := sorry

/-- A predicate that checks if a natural number has exactly 10 digits -/
def has_ten_digits (n : ℕ) : Prop := sorry

theorem max_odd_digits_in_sum (a b c : ℕ) 
  (ha : has_ten_digits a) 
  (hb : has_ten_digits b) 
  (hc : has_ten_digits c) 
  (sum_eq : a + b = c) : 
  count_odd_digits a + count_odd_digits b + count_odd_digits c ≤ 29 :=
sorry

end NUMINAMATH_CALUDE_max_odd_digits_in_sum_l538_53899


namespace NUMINAMATH_CALUDE_smallest_k_with_remainder_one_l538_53866

theorem smallest_k_with_remainder_one (k : ℕ) : k = 400 ↔ 
  (k > 1 ∧ 
   k % 19 = 1 ∧ 
   k % 7 = 1 ∧ 
   k % 3 = 1 ∧ 
   ∀ m : ℕ, m > 1 ∧ m % 19 = 1 ∧ m % 7 = 1 ∧ m % 3 = 1 → k ≤ m) :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_with_remainder_one_l538_53866


namespace NUMINAMATH_CALUDE_hyperbola_equation_l538_53854

/-- Hyperbola with center at origin, focus at (3,0), and intersection points with midpoint (-12,-15) -/
def Hyperbola (E : Set (ℝ × ℝ)) : Prop :=
  ∃ (A B : ℝ × ℝ) (l : Set (ℝ × ℝ)),
    (0, 0) ∈ E ∧  -- Center at origin
    (3, 0) ∈ E ∧  -- Focus at (3,0)
    (A ∈ E ∧ B ∈ E) ∧  -- A and B are on the hyperbola
    (A ∈ l ∧ B ∈ l ∧ (3, 0) ∈ l) ∧  -- A, B, and focus are on line l
    ((A.1 + B.1) / 2 = -12 ∧ (A.2 + B.2) / 2 = -15)  -- Midpoint of A and B is (-12,-15)

/-- The equation of the hyperbola -/
def HyperbolaEquation (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 5 = 1

theorem hyperbola_equation (E : Set (ℝ × ℝ)) (h : Hyperbola E) :
  ∀ (x y : ℝ), (x, y) ∈ E ↔ HyperbolaEquation x y := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l538_53854


namespace NUMINAMATH_CALUDE_reflection_line_equation_l538_53895

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  p : Point
  q : Point
  r : Point

/-- The reflection of a triangle -/
structure ReflectedTriangle where
  p' : Point
  q' : Point
  r' : Point

/-- The line of reflection -/
structure ReflectionLine where
  equation : ℝ → Prop

/-- Theorem: Given a triangle and its reflection, prove the equation of the reflection line -/
theorem reflection_line_equation 
  (t : Triangle) 
  (rt : ReflectedTriangle) 
  (h1 : t.p = ⟨2, 2⟩) 
  (h2 : t.q = ⟨6, 6⟩) 
  (h3 : t.r = ⟨-3, 5⟩)
  (h4 : rt.p' = ⟨2, -4⟩) 
  (h5 : rt.q' = ⟨6, -8⟩) 
  (h6 : rt.r' = ⟨-3, -7⟩) :
  ∃ (l : ReflectionLine), l.equation = λ y => y = -1 := by
  sorry

end NUMINAMATH_CALUDE_reflection_line_equation_l538_53895


namespace NUMINAMATH_CALUDE_intersection_point_power_l538_53888

theorem intersection_point_power (n : ℕ) (x₀ y₀ : ℝ) (hn : n ≥ 2) 
  (h1 : y₀^2 = n * x₀ - 1) (h2 : y₀ = x₀) :
  ∀ m : ℕ, m > 0 → ∃ k : ℕ, k ≥ 2 ∧ (x₀^m)^2 = k * (x₀^m) - 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_power_l538_53888


namespace NUMINAMATH_CALUDE_gcd_property_l538_53844

theorem gcd_property (n : ℤ) : 
  (∃ k : ℤ, n = 31 * k - 11) ↔ Int.gcd (5 * n - 7) (3 * n + 2) = 31 := by
  sorry

end NUMINAMATH_CALUDE_gcd_property_l538_53844


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_l538_53830

/-- The greatest common factor of 18, 30, and 45 -/
def C : ℕ := Nat.gcd 18 (Nat.gcd 30 45)

/-- The least common multiple of 18, 30, and 45 -/
def D : ℕ := Nat.lcm 18 (Nat.lcm 30 45)

/-- The sum of the greatest common factor and the least common multiple of 18, 30, and 45 is 93 -/
theorem gcd_lcm_sum : C + D = 93 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_l538_53830


namespace NUMINAMATH_CALUDE_sum_opposite_angles_inscribed_quadrilateral_l538_53849

/-- A quadrilateral WXYZ inscribed in a circle -/
structure InscribedQuadrilateral where
  /-- The measure of the angle subtended by arc WZ at the circumference -/
  angle_WZ : ℝ
  /-- The measure of the angle subtended by arc XY at the circumference -/
  angle_XY : ℝ

/-- Theorem: Sum of opposite angles in an inscribed quadrilateral -/
theorem sum_opposite_angles_inscribed_quadrilateral 
  (quad : InscribedQuadrilateral) 
  (h1 : quad.angle_WZ = 40)
  (h2 : quad.angle_XY = 20) :
  ∃ (angle_WXY angle_WZY : ℝ), angle_WXY + angle_WZY = 120 :=
sorry

end NUMINAMATH_CALUDE_sum_opposite_angles_inscribed_quadrilateral_l538_53849


namespace NUMINAMATH_CALUDE_trapezoid_longest_diagonal_lower_bound_trapezoid_longest_diagonal_lower_bound_tight_l538_53859

/-- A trapezoid with area 1 -/
structure Trapezoid :=
  (a b h : ℝ)  -- lengths of bases and height
  (d₁ d₂ : ℝ)  -- lengths of diagonals
  (area_eq : (a + b) * h / 2 = 1)
  (d₁_ge_d₂ : d₁ ≥ d₂)

/-- The longest diagonal of a trapezoid with area 1 is at least √2 -/
theorem trapezoid_longest_diagonal_lower_bound (T : Trapezoid) : 
  T.d₁ ≥ Real.sqrt 2 := by sorry

/-- There exists a trapezoid with area 1 whose longest diagonal is exactly √2 -/
theorem trapezoid_longest_diagonal_lower_bound_tight : 
  ∃ T : Trapezoid, T.d₁ = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_trapezoid_longest_diagonal_lower_bound_trapezoid_longest_diagonal_lower_bound_tight_l538_53859


namespace NUMINAMATH_CALUDE_ae_be_implies_p_or_q_l538_53817

theorem ae_be_implies_p_or_q (a b : ℝ) (h1 : a ≠ b) (h2 : a * Real.exp a = b * Real.exp b) :
  (Real.log a + a = Real.log b + b) ∨ ((a + 1) * (b + 1) < 0) := by
  sorry

end NUMINAMATH_CALUDE_ae_be_implies_p_or_q_l538_53817


namespace NUMINAMATH_CALUDE_fish_pond_flowers_l538_53893

/-- Calculates the number of flowers planted around a circular pond -/
def flowers_around_pond (perimeter : ℕ) (tree_spacing : ℕ) (flowers_between : ℕ) : ℕ :=
  (perimeter / tree_spacing) * flowers_between

/-- Theorem: The number of flowers planted around the fish pond is 39 -/
theorem fish_pond_flowers :
  flowers_around_pond 52 4 3 = 39 := by
  sorry

end NUMINAMATH_CALUDE_fish_pond_flowers_l538_53893


namespace NUMINAMATH_CALUDE_Q_sufficient_not_necessary_l538_53827

open Real

-- Define a differentiable function f on ℝ
variable (f : ℝ → ℝ)
variable (hf : Differentiable ℝ f)

-- Define proposition P
def P (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → |(f x₁ - f x₂) / (x₁ - x₂)| < 2018

-- Define proposition Q
def Q (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, |deriv f x| < 2018

-- Theorem stating that Q is sufficient but not necessary for P
theorem Q_sufficient_not_necessary (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  (Q f → P f) ∧ ∃ g : ℝ → ℝ, Differentiable ℝ g ∧ P g ∧ ¬(Q g) := by
  sorry

end NUMINAMATH_CALUDE_Q_sufficient_not_necessary_l538_53827


namespace NUMINAMATH_CALUDE_problem_statement_l538_53872

theorem problem_statement (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : y > x) 
  (h : x / y + y / x = 3) : (x + y) / (x - y) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l538_53872


namespace NUMINAMATH_CALUDE_asphalt_cost_asphalt_cost_proof_l538_53889

/-- Calculates the total cost of asphalt for paving a road, including sales tax. -/
theorem asphalt_cost (road_length : ℝ) (road_width : ℝ) (coverage_per_truckload : ℝ) 
  (cost_per_truckload : ℝ) (sales_tax_rate : ℝ) : ℝ :=
  let road_area := road_length * road_width
  let num_truckloads := road_area / coverage_per_truckload
  let total_cost_before_tax := num_truckloads * cost_per_truckload
  let sales_tax := total_cost_before_tax * sales_tax_rate
  let total_cost_with_tax := total_cost_before_tax + sales_tax
  total_cost_with_tax

/-- Proves that the total cost of asphalt for the given road specifications is $4,500. -/
theorem asphalt_cost_proof :
  asphalt_cost 2000 20 800 75 0.2 = 4500 := by
  sorry

end NUMINAMATH_CALUDE_asphalt_cost_asphalt_cost_proof_l538_53889


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l538_53804

theorem sqrt_equation_solution (x : ℝ) :
  (Real.sqrt x + Real.sqrt 243) / Real.sqrt 75 = 2.4 → x = 27 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l538_53804


namespace NUMINAMATH_CALUDE_common_root_equations_l538_53862

theorem common_root_equations (k : ℝ) :
  (∃ x : ℝ, x^2 - k*x - 7 = 0 ∧ x^2 - 6*x - (k + 1) = 0) →
  (k = -6 ∧
   (∃ x : ℝ, x^2 + 6*x - 7 = 0 ∧ x^2 - 6*x + 5 = 0 ∧ x = 1) ∧
   (∃ y z : ℝ, y^2 + 6*y - 7 = 0 ∧ z^2 - 6*z + 5 = 0 ∧ y = -7 ∧ z = 5)) :=
by sorry

end NUMINAMATH_CALUDE_common_root_equations_l538_53862


namespace NUMINAMATH_CALUDE_ellen_painting_time_l538_53802

/-- The time it takes Ellen to paint all flowers and vines -/
def total_painting_time (lily_time rose_time orchid_time vine_time : ℕ) 
                        (lily_count rose_count orchid_count vine_count : ℕ) : ℕ :=
  lily_time * lily_count + rose_time * rose_count + 
  orchid_time * orchid_count + vine_time * vine_count

/-- Theorem stating that Ellen's total painting time is 213 minutes -/
theorem ellen_painting_time : 
  total_painting_time 5 7 3 2 17 10 6 20 = 213 := by
  sorry

end NUMINAMATH_CALUDE_ellen_painting_time_l538_53802


namespace NUMINAMATH_CALUDE_prob_theorem_l538_53816

/-- Represents the colors of the balls in the bag -/
inductive Color
  | Black
  | Yellow
  | Green

/-- The total number of balls in the bag -/
def total_balls : ℕ := 9

/-- The probability of drawing either a black or a yellow ball -/
def prob_black_or_yellow : ℚ := 5/9

/-- The probability of drawing either a yellow or a green ball -/
def prob_yellow_or_green : ℚ := 2/3

/-- The probability of drawing a ball of a specific color -/
def prob_color (c : Color) : ℚ :=
  match c with
  | Color.Black => 1/3
  | Color.Yellow => 2/9
  | Color.Green => 4/9

/-- The probability of drawing two balls of different colors -/
def prob_different_colors : ℚ := 13/18

/-- Theorem stating the probabilities are correct given the conditions -/
theorem prob_theorem :
  (∀ c, prob_color c ≥ 0) ∧
  (prob_color Color.Black + prob_color Color.Yellow + prob_color Color.Green = 1) ∧
  (prob_color Color.Black + prob_color Color.Yellow = prob_black_or_yellow) ∧
  (prob_color Color.Yellow + prob_color Color.Green = prob_yellow_or_green) ∧
  (prob_different_colors = 13/18) :=
  sorry

end NUMINAMATH_CALUDE_prob_theorem_l538_53816


namespace NUMINAMATH_CALUDE_fourth_root_of_105413504_l538_53842

theorem fourth_root_of_105413504 : (105413504 : ℝ) ^ (1/4 : ℝ) = 101 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_of_105413504_l538_53842


namespace NUMINAMATH_CALUDE_min_x_prime_factorization_l538_53897

theorem min_x_prime_factorization (x y : ℕ+) (h : 5 * x^7 = 13 * y^11) :
  ∃ (a b c d : ℕ), 
    (x = a^c * b^d) ∧ 
    (x ≥ 13^6 * 5^7) ∧
    (x = 13^6 * 5^7 → a + b + c + d = 31) :=
sorry

end NUMINAMATH_CALUDE_min_x_prime_factorization_l538_53897


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l538_53848

/-- An arithmetic sequence with its sum sequence -/
structure ArithmeticSequence :=
  (a : ℕ → ℝ)  -- The sequence
  (S : ℕ → ℝ)  -- The sum sequence
  (h1 : ∀ n, S n = (n : ℝ) / 2 * (2 * a 1 + (n - 1) * (a 2 - a 1)))  -- Definition of sum
  (h2 : ∀ n, a (n + 1) = a n + (a 2 - a 1))  -- Definition of arithmetic sequence

/-- The main theorem -/
theorem arithmetic_sequence_fifth_term 
  (seq : ArithmeticSequence) 
  (eq1 : seq.a 3 + seq.S 3 = 22) 
  (eq2 : seq.a 4 - seq.S 4 = -15) : 
  seq.a 5 = 11 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l538_53848


namespace NUMINAMATH_CALUDE_expression_evaluation_l538_53807

theorem expression_evaluation : (2 + 6 * 3 - 4) + 2^3 * 4 / 2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l538_53807


namespace NUMINAMATH_CALUDE_floor_difference_equals_five_l538_53839

theorem floor_difference_equals_five (n : ℤ) : 
  (Int.floor (n^2 / 4 : ℚ) - Int.floor (n / 2 : ℚ)^2 = 5) → n = 11 := by
  sorry

end NUMINAMATH_CALUDE_floor_difference_equals_five_l538_53839


namespace NUMINAMATH_CALUDE_min_product_a_purchase_l538_53846

theorem min_product_a_purchase (cost_a cost_b total_items max_cost : ℕ) 
  (h1 : cost_a = 20)
  (h2 : cost_b = 50)
  (h3 : total_items = 10)
  (h4 : max_cost = 350) : 
  ∃ min_a : ℕ, min_a = 5 ∧ 
  ∀ x : ℕ, (x ≤ total_items ∧ x * cost_a + (total_items - x) * cost_b ≤ max_cost) → x ≥ min_a := by
  sorry

end NUMINAMATH_CALUDE_min_product_a_purchase_l538_53846


namespace NUMINAMATH_CALUDE_repairs_count_l538_53840

/-- Represents the mechanic shop scenario --/
structure MechanicShop where
  oil_change_price : ℕ
  repair_price : ℕ
  car_wash_price : ℕ
  oil_changes : ℕ
  car_washes : ℕ
  total_earnings : ℕ

/-- Calculates the number of repairs given the shop's data --/
def calculate_repairs (shop : MechanicShop) : ℕ :=
  (shop.total_earnings - (shop.oil_change_price * shop.oil_changes + shop.car_wash_price * shop.car_washes)) / shop.repair_price

/-- Theorem stating that given the specific conditions, the number of repairs is 10 --/
theorem repairs_count (shop : MechanicShop) 
  (h1 : shop.oil_change_price = 20)
  (h2 : shop.repair_price = 30)
  (h3 : shop.car_wash_price = 5)
  (h4 : shop.oil_changes = 5)
  (h5 : shop.car_washes = 15)
  (h6 : shop.total_earnings = 475) :
  calculate_repairs shop = 10 := by
  sorry

#eval calculate_repairs { 
  oil_change_price := 20, 
  repair_price := 30, 
  car_wash_price := 5, 
  oil_changes := 5, 
  car_washes := 15, 
  total_earnings := 475 
}

end NUMINAMATH_CALUDE_repairs_count_l538_53840


namespace NUMINAMATH_CALUDE_school_children_count_l538_53811

theorem school_children_count : 
  ∀ (B C : ℕ), 
    B = 2 * C → 
    B = 4 * (C - 370) → 
    C = 740 := by
  sorry

end NUMINAMATH_CALUDE_school_children_count_l538_53811


namespace NUMINAMATH_CALUDE_inequality_proof_l538_53820

def M : Set ℝ := {x | |x + 1| + |x - 1| ≤ 2}

theorem inequality_proof (x y z : ℝ) (hx : x ∈ M) (hy : |y| ≤ 1/6) (hz : |z| ≤ 1/9) :
  |x + 2*y - 3*z| ≤ 5/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l538_53820


namespace NUMINAMATH_CALUDE_f_strictly_increasing_l538_53864

def f (x : ℝ) : ℝ := x^3 - x^2 - x

theorem f_strictly_increasing :
  (∀ x y, x < y ∧ x < -1/3 → f x < f y) ∧
  (∀ x y, x < y ∧ 1 < x → f x < f y) :=
sorry

end NUMINAMATH_CALUDE_f_strictly_increasing_l538_53864


namespace NUMINAMATH_CALUDE_smallest_sum_20_consecutive_triangular_l538_53819

/-- The sum of 20 consecutive integers starting from n -/
def sum_20_consecutive (n : ℤ) : ℤ := 10 * (2 * n + 19)

/-- A triangular number -/
def triangular_number (m : ℕ) : ℕ := m * (m + 1) / 2

/-- Proposition: 190 is the smallest sum of 20 consecutive integers that is also a triangular number -/
theorem smallest_sum_20_consecutive_triangular :
  ∃ (m : ℕ), 
    (∀ (n : ℤ), sum_20_consecutive n ≥ 190) ∧ 
    (sum_20_consecutive 0 = 190) ∧ 
    (triangular_number m = 190) :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_20_consecutive_triangular_l538_53819


namespace NUMINAMATH_CALUDE_x1_range_proof_l538_53860

theorem x1_range_proof (f : ℝ → ℝ) (h_incr : Monotone f) :
  (∀ x₁ x₂ : ℝ, x₁ + x₂ = 1 → f x₁ + f 0 > f x₂ + f 1) →
  ∀ x₁ : ℝ, (∃ x₂ : ℝ, x₁ + x₂ = 1 ∧ f x₁ + f 0 > f x₂ + f 1) → x₁ > 1 :=
by sorry

end NUMINAMATH_CALUDE_x1_range_proof_l538_53860


namespace NUMINAMATH_CALUDE_train_crossing_poles_time_l538_53847

/-- Calculates the total time for a train to cross multiple poles -/
theorem train_crossing_poles_time
  (train_speed : ℝ)
  (first_pole_crossing_time : ℝ)
  (pole_distances : List ℝ)
  (h1 : train_speed = 75)  -- 75 kmph
  (h2 : first_pole_crossing_time = 3)  -- 3 seconds
  (h3 : pole_distances = [500, 800, 1500, 2200]) :  -- distances in meters
  ∃ (total_time : ℝ),
    total_time = 243 ∧  -- 243 seconds
    total_time = first_pole_crossing_time +
      (pole_distances.map (λ d => d / (train_speed * 1000 / 3600))).sum :=
by sorry

end NUMINAMATH_CALUDE_train_crossing_poles_time_l538_53847


namespace NUMINAMATH_CALUDE_tetrahedron_inequality_l538_53856

theorem tetrahedron_inequality (a b c d h_a h_b h_c h_d V : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (hV : V > 0) 
  (h1 : V = (1/3) * a * h_a) 
  (h2 : V = (1/3) * b * h_b) 
  (h3 : V = (1/3) * c * h_c) 
  (h4 : V = (1/3) * d * h_d) : 
  (a + b + c + d) * (h_a + h_b + h_c + h_d) ≥ 48 * V := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_inequality_l538_53856


namespace NUMINAMATH_CALUDE_total_liquid_consumed_l538_53883

/-- Proves that the total amount of liquid consumed by Yurim and Ji-in is 6300 milliliters -/
theorem total_liquid_consumed (yurim_liters : ℕ) (yurim_ml : ℕ) (jiin_ml : ℕ) :
  yurim_liters = 2 →
  yurim_ml = 600 →
  jiin_ml = 3700 →
  yurim_liters * 1000 + yurim_ml + jiin_ml = 6300 :=
by
  sorry

end NUMINAMATH_CALUDE_total_liquid_consumed_l538_53883


namespace NUMINAMATH_CALUDE_polynomial_expansion_l538_53810

theorem polynomial_expansion (t : ℝ) :
  (3 * t^3 - 2 * t^2 + t - 4) * (4 * t^2 - 2 * t + 5) =
  12 * t^5 - 14 * t^4 + 23 * t^3 - 28 * t^2 + 13 * t - 20 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l538_53810


namespace NUMINAMATH_CALUDE_negative_integer_sum_and_square_equals_neg_twelve_l538_53863

theorem negative_integer_sum_and_square_equals_neg_twelve (N : ℤ) :
  N < 0 → N^2 + N = -12 → N = -3 ∨ N = -4 := by
  sorry

end NUMINAMATH_CALUDE_negative_integer_sum_and_square_equals_neg_twelve_l538_53863


namespace NUMINAMATH_CALUDE_tank_capacity_is_900_l538_53806

/-- Represents the capacity of a tank and its filling/draining rates. -/
structure TankSystem where
  capacity : ℕ
  fill_rate_A : ℕ
  fill_rate_B : ℕ
  drain_rate_C : ℕ

/-- Calculates the net amount of water added to the tank in one cycle. -/
def net_fill_per_cycle (t : TankSystem) : ℕ :=
  t.fill_rate_A + t.fill_rate_B - t.drain_rate_C

/-- Theorem stating that under given conditions, the tank capacity is 900 liters. -/
theorem tank_capacity_is_900 (t : TankSystem) 
  (h1 : t.fill_rate_A = 40)
  (h2 : t.fill_rate_B = 30)
  (h3 : t.drain_rate_C = 20)
  (h4 : (54 : ℕ) * (net_fill_per_cycle t) / 3 = t.capacity) :
  t.capacity = 900 := by
  sorry

#check tank_capacity_is_900

end NUMINAMATH_CALUDE_tank_capacity_is_900_l538_53806


namespace NUMINAMATH_CALUDE_parallelogram_count_l538_53826

/-- Given a triangle ABC with each side divided into n equal segments and connected by parallel lines,
    f(n) represents the total number of parallelograms formed within the network. -/
def f (n : ℕ) : ℕ := 3 * (Nat.choose (n + 2) 4)

/-- Theorem stating that f(n) correctly counts the number of parallelograms in the described configuration. -/
theorem parallelogram_count (n : ℕ) : 
  f n = 3 * (Nat.choose (n + 2) 4) := by sorry

end NUMINAMATH_CALUDE_parallelogram_count_l538_53826


namespace NUMINAMATH_CALUDE_problem_statement_l538_53875

theorem problem_statement (x y : ℝ) (h : |x + 2| + Real.sqrt (y - 3) = 0) :
  (x + y) ^ 2023 = 1 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l538_53875


namespace NUMINAMATH_CALUDE_min_amount_lost_l538_53825

/-- Represents the denomination of a bill -/
inductive Bill
  | ten  : Bill
  | fifty : Bill

/-- Calculates the value of a bill -/
def billValue (b : Bill) : Nat :=
  match b with
  | Bill.ten  => 10
  | Bill.fifty => 50

/-- Represents the cash transaction and usage -/
structure CashTransaction where
  totalCashed : Nat
  billsUsed : Nat
  tenBills : Nat
  fiftyBills : Nat

/-- Conditions of the problem -/
def transactionConditions (t : CashTransaction) : Prop :=
  t.totalCashed = 1270 ∧
  t.billsUsed = 15 ∧
  (t.tenBills = t.fiftyBills + 1 ∨ t.tenBills = t.fiftyBills - 1) ∧
  t.tenBills * billValue Bill.ten + t.fiftyBills * billValue Bill.fifty ≤ t.totalCashed

/-- Theorem stating the minimum amount lost -/
theorem min_amount_lost (t : CashTransaction) 
  (h : transactionConditions t) : 
  t.totalCashed - (t.tenBills * billValue Bill.ten + t.fiftyBills * billValue Bill.fifty) = 800 := by
  sorry

end NUMINAMATH_CALUDE_min_amount_lost_l538_53825


namespace NUMINAMATH_CALUDE_circle_area_with_diameter_9_l538_53870

theorem circle_area_with_diameter_9 (π : Real) (h : π = Real.pi) :
  let d := 9
  let r := d / 2
  let area := π * r^2
  area = π * (9/2)^2 := by sorry

end NUMINAMATH_CALUDE_circle_area_with_diameter_9_l538_53870


namespace NUMINAMATH_CALUDE_largest_inscribed_triangle_area_l538_53885

theorem largest_inscribed_triangle_area (r : ℝ) (h : r = 8) :
  let circle_area := π * r^2
  let diameter := 2 * r
  let max_height := r
  let triangle_area := (1/2) * diameter * max_height
  triangle_area = 64 := by sorry

end NUMINAMATH_CALUDE_largest_inscribed_triangle_area_l538_53885


namespace NUMINAMATH_CALUDE_only_negative_three_less_than_reciprocal_l538_53871

def is_less_than_reciprocal (x : ℝ) : Prop :=
  x ≠ 0 ∧ x < 1 / x

theorem only_negative_three_less_than_reciprocal :
  (is_less_than_reciprocal (-3)) ∧
  (¬ is_less_than_reciprocal (-1/2)) ∧
  (¬ is_less_than_reciprocal 0) ∧
  (¬ is_less_than_reciprocal 1) ∧
  (¬ is_less_than_reciprocal (3/2)) :=
by sorry

end NUMINAMATH_CALUDE_only_negative_three_less_than_reciprocal_l538_53871


namespace NUMINAMATH_CALUDE_a_51_value_l538_53823

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, a (n + 1) - a n = 2

theorem a_51_value (a : ℕ → ℤ) (h : arithmetic_sequence a) : a 51 = 101 := by
  sorry

end NUMINAMATH_CALUDE_a_51_value_l538_53823


namespace NUMINAMATH_CALUDE_quadratic_solution_l538_53822

theorem quadratic_solution : ∃ x₁ x₂ : ℝ, x₁ = 6 ∧ x₂ = -1 ∧ ∀ x : ℝ, x^2 - 5*x - 6 = 0 ↔ x = x₁ ∨ x = x₂ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_l538_53822


namespace NUMINAMATH_CALUDE_roy_tablet_interval_l538_53874

/-- Given a total number of tablets and the total time to consume them,
    calculate the time interval between each tablet. -/
def tablet_interval (num_tablets : ℕ) (total_time : ℕ) : ℕ :=
  total_time / (num_tablets - 1)

/-- Theorem stating that for 5 tablets consumed over 60 minutes,
    the time interval between each tablet is 15 minutes. -/
theorem roy_tablet_interval :
  tablet_interval 5 60 = 15 := by
  sorry

#eval tablet_interval 5 60

end NUMINAMATH_CALUDE_roy_tablet_interval_l538_53874


namespace NUMINAMATH_CALUDE_unique_triple_gcd_sum_l538_53834

theorem unique_triple_gcd_sum (m n l : ℕ) : 
  (m + n = (Nat.gcd m n)^2) ∧ 
  (m + l = (Nat.gcd m l)^2) ∧ 
  (n + l = (Nat.gcd n l)^2) →
  m = 2 ∧ n = 2 ∧ l = 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_triple_gcd_sum_l538_53834


namespace NUMINAMATH_CALUDE_quadratic_root_sum_product_l538_53896

theorem quadratic_root_sum_product (m n : ℝ) : 
  (∃ x y : ℝ, 3 * x^2 - m * x + n = 0 ∧ 3 * y^2 - m * y + n = 0 ∧ x + y = 9 ∧ x * y = 14) → 
  m + n = 69 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_product_l538_53896


namespace NUMINAMATH_CALUDE_certain_number_value_l538_53813

theorem certain_number_value (t b c : ℝ) : 
  (t + b + c + 14 + 15) / 5 = 12 → 
  ∃ x : ℝ, (t + b + c + x) / 4 = 15 ∧ x = 29 := by
sorry

end NUMINAMATH_CALUDE_certain_number_value_l538_53813


namespace NUMINAMATH_CALUDE_systematic_sampling_l538_53828

theorem systematic_sampling (total_students : Nat) (num_groups : Nat) (selected_in_first_group : Nat) (target_group : Nat) : 
  total_students = 480 →
  num_groups = 30 →
  selected_in_first_group = 5 →
  target_group = 8 →
  (total_students / num_groups) * (target_group - 1) + selected_in_first_group = 117 :=
by
  sorry

#check systematic_sampling

end NUMINAMATH_CALUDE_systematic_sampling_l538_53828


namespace NUMINAMATH_CALUDE_incorrect_expression_l538_53818

def repeating_decimal (X Y Z : ℕ) (t u v : ℕ) : ℚ :=
  sorry

theorem incorrect_expression
  (E : ℚ) (X Y Z : ℕ) (t u v : ℕ)
  (h_E : E = repeating_decimal X Y Z t u v) :
  ¬(10^t * (10^(u+v) - 1) * E = Z * (Y - 1)) :=
sorry

end NUMINAMATH_CALUDE_incorrect_expression_l538_53818


namespace NUMINAMATH_CALUDE_roots_sum_product_l538_53898

theorem roots_sum_product (a b : ℂ) : 
  (a ≠ b) → 
  (a^3 + 3*a^2 + a + 1 = 0) → 
  (b^3 + 3*b^2 + b + 1 = 0) → 
  (a^2 * b + a * b^2 + 3*a*b = 1) := by
sorry

end NUMINAMATH_CALUDE_roots_sum_product_l538_53898


namespace NUMINAMATH_CALUDE_unknown_blanket_rate_l538_53845

theorem unknown_blanket_rate (blanket_price_1 blanket_price_2 average_price : ℚ)
  (num_blankets_1 num_blankets_2 num_blankets_unknown : ℕ) :
  blanket_price_1 = 100 →
  blanket_price_2 = 150 →
  num_blankets_1 = 2 →
  num_blankets_2 = 5 →
  num_blankets_unknown = 2 →
  average_price = 150 →
  ∃ unknown_price : ℚ,
    (num_blankets_1 * blanket_price_1 + num_blankets_2 * blanket_price_2 + num_blankets_unknown * unknown_price) /
    (num_blankets_1 + num_blankets_2 + num_blankets_unknown) = average_price ∧
    unknown_price = 200 :=
by sorry

end NUMINAMATH_CALUDE_unknown_blanket_rate_l538_53845


namespace NUMINAMATH_CALUDE_square_sum_value_l538_53831

theorem square_sum_value (x y : ℝ) (h1 : (x + y)^2 = 36) (h2 : x * y = 8) :
  x^2 + y^2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_value_l538_53831


namespace NUMINAMATH_CALUDE_lowest_sale_price_percentage_l538_53876

theorem lowest_sale_price_percentage (list_price : ℝ) (max_discount : ℝ) (additional_discount : ℝ) :
  list_price = 80 →
  max_discount = 0.5 →
  additional_discount = 0.2 →
  let discounted_price := list_price * (1 - max_discount)
  let final_price := discounted_price - (list_price * additional_discount)
  final_price / list_price = 0.3 := by sorry

end NUMINAMATH_CALUDE_lowest_sale_price_percentage_l538_53876


namespace NUMINAMATH_CALUDE_log_ratio_equals_two_thirds_l538_53857

theorem log_ratio_equals_two_thirds :
  (Real.log 9 / Real.log 8) / (Real.log 3 / Real.log 2) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_log_ratio_equals_two_thirds_l538_53857


namespace NUMINAMATH_CALUDE_cos_two_theta_value_l538_53815

theorem cos_two_theta_value (θ : Real) 
  (h : Real.sin (θ / 2) + Real.cos (θ / 2) = 2 * Real.sqrt 2 / 3) : 
  Real.cos (2 * θ) = 79 / 81 := by
  sorry

end NUMINAMATH_CALUDE_cos_two_theta_value_l538_53815


namespace NUMINAMATH_CALUDE_three_numbers_sum_l538_53881

theorem three_numbers_sum (a b c : ℝ) : 
  a ≤ b → b ≤ c → 
  b = 10 → 
  (a + b + c) / 3 = a + 15 → 
  (a + b + c) / 3 = c - 25 → 
  a + b + c = 60 := by
sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l538_53881


namespace NUMINAMATH_CALUDE_cone_generatrix_length_l538_53833

/-- The length of the generatrix of a cone with lateral area 6π and base radius 2 is 3. -/
theorem cone_generatrix_length :
  ∀ (l : ℝ), 
    (l > 0) →
    (2 * Real.pi * l = 6 * Real.pi) →
    l = 3 := by
  sorry

end NUMINAMATH_CALUDE_cone_generatrix_length_l538_53833


namespace NUMINAMATH_CALUDE_sunny_cake_candles_l538_53837

/-- Given the initial number of cakes, number of cakes given away, and total candles used,
    calculate the number of candles on each remaining cake. -/
def candles_per_cake (initial_cakes : ℕ) (cakes_given_away : ℕ) (total_candles : ℕ) : ℕ :=
  total_candles / (initial_cakes - cakes_given_away)

/-- Prove that given the specific values in the problem, 
    the number of candles on each remaining cake is 6. -/
theorem sunny_cake_candles : 
  candles_per_cake 8 2 36 = 6 := by sorry

end NUMINAMATH_CALUDE_sunny_cake_candles_l538_53837


namespace NUMINAMATH_CALUDE_valid_paintings_count_l538_53869

/-- Represents a color in the painting. -/
inductive Color
  | Green
  | Red
  | Blue

/-- Represents a position in the 3x3 grid. -/
structure Position :=
  (row : Fin 3)
  (col : Fin 3)

/-- Represents a painting of the 3x3 grid. -/
def Painting := Position → Color

/-- Checks if a painting satisfies the color placement rules. -/
def validPainting (p : Painting) : Prop :=
  ∀ (pos : Position),
    (p pos = Color.Green →
      ∀ (above : Position), above.row = pos.row - 1 → above.col = pos.col → p above ≠ Color.Red) ∧
    (p pos = Color.Green →
      ∀ (right : Position), right.row = pos.row → right.col = pos.col + 1 → p right ≠ Color.Red) ∧
    (p pos = Color.Blue →
      ∀ (left : Position), left.row = pos.row → left.col = pos.col - 1 → p left ≠ Color.Red)

/-- The number of valid paintings. -/
def numValidPaintings : ℕ := sorry

theorem valid_paintings_count :
  numValidPaintings = 78 :=
sorry

end NUMINAMATH_CALUDE_valid_paintings_count_l538_53869


namespace NUMINAMATH_CALUDE_segments_form_triangle_l538_53803

/-- Triangle inequality theorem: the sum of the lengths of any two sides 
    of a triangle must be greater than the length of the remaining side -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Function to check if three line segments can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

/-- Theorem stating that line segments of lengths 13, 12, and 20 can form a triangle -/
theorem segments_form_triangle : can_form_triangle 13 12 20 := by
  sorry


end NUMINAMATH_CALUDE_segments_form_triangle_l538_53803


namespace NUMINAMATH_CALUDE_min_value_of_quadratic_l538_53890

/-- The function f(x) = x^2 - 2x + 3 has a minimum value of 2 for positive x -/
theorem min_value_of_quadratic (x : ℝ) (h : x > 0) :
  ∃ (min_val : ℝ), min_val = 2 ∧ ∀ y, y > 0 → x^2 - 2*x + 3 ≥ min_val := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_quadratic_l538_53890


namespace NUMINAMATH_CALUDE_bill_bathroom_visits_l538_53882

/-- The number of times Bill goes to the bathroom daily -/
def bathroom_visits : ℕ := 3

/-- The number of squares of toilet paper Bill uses per bathroom visit -/
def squares_per_visit : ℕ := 5

/-- The number of rolls of toilet paper Bill has -/
def total_rolls : ℕ := 1000

/-- The number of squares of toilet paper per roll -/
def squares_per_roll : ℕ := 300

/-- The number of days Bill's toilet paper supply will last -/
def supply_duration : ℕ := 20000

theorem bill_bathroom_visits :
  bathroom_visits * squares_per_visit * supply_duration = total_rolls * squares_per_roll := by
  sorry

end NUMINAMATH_CALUDE_bill_bathroom_visits_l538_53882


namespace NUMINAMATH_CALUDE_project_work_time_difference_l538_53891

/-- Given three people working on a project with their working times in the ratio of 3:5:6,
    and a total project time of 140 hours, prove that the difference between the working hours
    of the person who worked the most and the person who worked the least is 30 hours. -/
theorem project_work_time_difference :
  ∀ (x : ℝ), 
  (3 * x + 5 * x + 6 * x = 140) →
  (6 * x - 3 * x = 30) :=
by sorry

end NUMINAMATH_CALUDE_project_work_time_difference_l538_53891


namespace NUMINAMATH_CALUDE_negative_one_third_squared_l538_53880

theorem negative_one_third_squared : (-1/3 : ℚ)^2 = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_negative_one_third_squared_l538_53880


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l538_53861

def A : Set ℝ := {x | x^2 < 4}
def B : Set ℝ := {y | ∃ x ∈ A, y = x^2 - 2*x - 1}

theorem union_of_A_and_B : A ∪ B = {x | -2 ≤ x ∧ x < 7} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l538_53861


namespace NUMINAMATH_CALUDE_inverse_g_sum_l538_53877

-- Define the function g
def g (x : ℝ) : ℝ := x * |x|^3

-- State the theorem
theorem inverse_g_sum : g⁻¹ 8 + g⁻¹ (-64) = -2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_g_sum_l538_53877


namespace NUMINAMATH_CALUDE_triangle_ABC_properties_l538_53812

theorem triangle_ABC_properties (A B C : Real) (a b c : Real) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  b = 4 →
  (Real.cos B) / (Real.cos C) = 4 / (2 * a - c) →
  -- Conditions for a valid triangle
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  -- Prove the following
  B = π / 3 ∧
  (∀ S : Real, S = (1/2) * a * c * Real.sin B → S ≤ 4 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_ABC_properties_l538_53812


namespace NUMINAMATH_CALUDE_couple_driving_exam_probability_l538_53887

/-- Represents the probability of passing an exam for each attempt -/
structure ExamProbability where
  male : ℚ
  female : ℚ

/-- Represents the exam attempt limits and fee structure -/
structure ExamRules where
  free_attempts : ℕ
  max_attempts : ℕ
  fee : ℚ

/-- Calculates the probability of a couple passing the exam under given conditions -/
def couple_exam_probability (prob : ExamProbability) (rules : ExamRules) : ℚ × ℚ :=
  sorry

theorem couple_driving_exam_probability :
  let prob := ExamProbability.mk (3/4) (2/3)
  let rules := ExamRules.mk 2 5 200
  let result := couple_exam_probability prob rules
  result.1 = 5/6 ∧ result.2 = 1/9 :=
sorry

end NUMINAMATH_CALUDE_couple_driving_exam_probability_l538_53887


namespace NUMINAMATH_CALUDE_valid_student_totals_l538_53850

/-- Represents the distribution of students in groups -/
structure StudentDistribution where
  total_groups : Nat
  groups_with_13 : Nat
  total_students : Nat

/-- Checks if a given distribution is valid according to the problem conditions -/
def is_valid_distribution (d : StudentDistribution) : Prop :=
  d.total_groups = 6 ∧
  d.groups_with_13 = 4 ∧
  (d.total_students = 76 ∨ d.total_students = 80)

/-- Theorem stating that the only valid total numbers of students are 76 and 80 -/
theorem valid_student_totals :
  ∀ d : StudentDistribution,
    is_valid_distribution d →
    (d.total_students = 76 ∨ d.total_students = 80) :=
by
  sorry

#check valid_student_totals

end NUMINAMATH_CALUDE_valid_student_totals_l538_53850


namespace NUMINAMATH_CALUDE_trishas_walk_l538_53855

/-- Proves that given a total distance and two equal segments, the remaining distance is as expected. -/
theorem trishas_walk (total : ℝ) (segment : ℝ) (h1 : total = 0.8888888888888888) 
  (h2 : segment = 0.1111111111111111) : 
  total - 2 * segment = 0.6666666666666666 := by
  sorry

end NUMINAMATH_CALUDE_trishas_walk_l538_53855


namespace NUMINAMATH_CALUDE_f_6_equals_0_l538_53873

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x in ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- A function f: ℝ → ℝ has the property f(x+2) = -f(x) for all x in ℝ -/
def HasPeriod2WithSignFlip (f : ℝ → ℝ) : Prop := ∀ x, f (x + 2) = -f x

theorem f_6_equals_0 (f : ℝ → ℝ) (h1 : IsOdd f) (h2 : HasPeriod2WithSignFlip f) : f 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_6_equals_0_l538_53873


namespace NUMINAMATH_CALUDE_solution_value_l538_53894

theorem solution_value (a b : ℝ) (h : 2 * a - 3 * b - 1 = 0) : 5 - 4 * a + 6 * b = 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l538_53894


namespace NUMINAMATH_CALUDE_car_total_distance_l538_53879

/-- A car driving through a ring in a tunnel -/
structure CarInRing where
  /-- Number of right-hand turns in the ring -/
  turns : ℕ
  /-- Distance traveled after the 1st turn -/
  dist1 : ℝ
  /-- Distance traveled after the 2nd turn -/
  dist2 : ℝ
  /-- Distance traveled after the 3rd turn -/
  dist3 : ℝ

/-- The total distance driven by the car around the ring -/
def totalDistance (car : CarInRing) : ℝ :=
  car.dist1 + car.dist2 + car.dist3

/-- Theorem stating the total distance driven by the car -/
theorem car_total_distance (car : CarInRing) 
  (h1 : car.turns = 4)
  (h2 : car.dist1 = 5)
  (h3 : car.dist2 = 8)
  (h4 : car.dist3 = 10) : 
  totalDistance car = 23 := by
  sorry

end NUMINAMATH_CALUDE_car_total_distance_l538_53879


namespace NUMINAMATH_CALUDE_paco_salty_cookies_left_l538_53884

/-- The number of salty cookies Paco has left after sharing with friends -/
def salty_cookies_left (initial_salty : ℕ) (shared_ana : ℕ) (shared_juan : ℕ) : ℕ :=
  initial_salty - (shared_ana + shared_juan)

/-- Theorem stating that Paco has 12 salty cookies left -/
theorem paco_salty_cookies_left :
  salty_cookies_left 26 11 3 = 12 := by sorry

end NUMINAMATH_CALUDE_paco_salty_cookies_left_l538_53884


namespace NUMINAMATH_CALUDE_exist_point_W_l538_53892

-- Define the triangle XYZ
def Triangle (X Y Z : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := X
  let (x₂, y₂) := Y
  let (x₃, y₃) := Z
  (x₁ - x₂)^2 + (y₁ - y₂)^2 = 10^2 ∧
  (x₂ - x₃)^2 + (y₂ - y₃)^2 = 11^2 ∧
  (x₁ - x₃)^2 + (y₁ - y₃)^2 = 12^2

-- Define point P on XZ
def PointP (X Z P : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := X
  let (x₃, y₃) := Z
  let (xp, yp) := P
  (xp - x₃)^2 + (yp - y₃)^2 = 6^2 ∧
  ∃ t : ℝ, 0 < t ∧ t < 1 ∧ xp = t * x₁ + (1 - t) * x₃ ∧ yp = t * y₁ + (1 - t) * y₃

-- Define point W on line PY
def PointW (Y P W : ℝ × ℝ) : Prop :=
  let (x₂, y₂) := Y
  let (xp, yp) := P
  let (xw, yw) := W
  ∃ t : ℝ, xw = t * xp + (1 - t) * x₂ ∧ yw = t * yp + (1 - t) * y₂

-- Define XW parallel to ZY
def Parallel (X W Z Y : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := X
  let (xw, yw) := W
  let (x₃, y₃) := Z
  let (x₂, y₂) := Y
  (xw - x₁) * (y₃ - y₂) = (yw - y₁) * (x₃ - x₂)

-- Define cyclic hexagon
def CyclicHexagon (Y X Y Z W X : ℝ × ℝ) : Prop :=
  -- This is a simplified definition, as the full condition for a cyclic hexagon is complex
  -- In reality, we would need to check if all six points lie on a circle
  true

-- Main theorem
theorem exist_point_W (X Y Z P : ℝ × ℝ) :
  Triangle X Y Z →
  PointP X Z P →
  ∃ W : ℝ × ℝ,
    PointW Y P W ∧
    Parallel X W Z Y ∧
    CyclicHexagon Y X Y Z W X ∧
    let (xp, yp) := P
    let (xw, yw) := W
    (xw - xp)^2 + (yw - yp)^2 = 10^2 :=
by sorry

end NUMINAMATH_CALUDE_exist_point_W_l538_53892


namespace NUMINAMATH_CALUDE_fraction_equality_l538_53801

theorem fraction_equality (x : ℝ) (h : x = 5) : (x^4 - 8*x^2 + 16) / (x^2 - 4) = 21 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l538_53801


namespace NUMINAMATH_CALUDE_man_work_days_l538_53853

/-- Given a woman can complete a piece of work in 40 days and a man is 25% more efficient than a woman,
    prove that the man can complete the same piece of work in 32 days. -/
theorem man_work_days (woman_days : ℕ) (man_efficiency : ℚ) :
  woman_days = 40 →
  man_efficiency = 5 / 4 →
  ∃ (man_days : ℕ), man_days = 32 ∧ (man_days : ℚ) * man_efficiency = woman_days := by
  sorry

end NUMINAMATH_CALUDE_man_work_days_l538_53853


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l538_53808

theorem sufficient_not_necessary_condition (x : ℝ) : 
  (x = 1 → x^3 = x) ∧ ¬(x^3 = x → x = 1) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l538_53808


namespace NUMINAMATH_CALUDE_renatas_transactions_l538_53843

/-- Represents Renata's financial transactions and final balance --/
theorem renatas_transactions (initial_amount casino_and_water_cost lottery_win final_balance : ℚ) :
  initial_amount = 10 →
  lottery_win = 65 →
  final_balance = 94 →
  casino_and_water_cost = 67 →
  initial_amount - 4 + 90 - casino_and_water_cost + lottery_win = final_balance :=
by sorry

end NUMINAMATH_CALUDE_renatas_transactions_l538_53843


namespace NUMINAMATH_CALUDE_bus_driver_compensation_l538_53852

def regular_rate : ℝ := 16
def regular_hours : ℝ := 40
def overtime_rate_increase : ℝ := 0.75
def total_hours : ℝ := 54

def overtime_rate : ℝ := regular_rate * (1 + overtime_rate_increase)
def overtime_hours : ℝ := total_hours - regular_hours

def regular_pay : ℝ := regular_rate * regular_hours
def overtime_pay : ℝ := overtime_rate * overtime_hours
def total_compensation : ℝ := regular_pay + overtime_pay

theorem bus_driver_compensation :
  total_compensation = 1032 := by sorry

end NUMINAMATH_CALUDE_bus_driver_compensation_l538_53852


namespace NUMINAMATH_CALUDE_molar_mass_calculation_l538_53814

/-- Given a chemical compound where 10 moles weigh 2070 grams, 
    prove that its molar mass is 207 grams/mole. -/
theorem molar_mass_calculation (total_weight : ℝ) (num_moles : ℝ) 
  (h1 : total_weight = 2070)
  (h2 : num_moles = 10) :
  total_weight / num_moles = 207 := by
  sorry

end NUMINAMATH_CALUDE_molar_mass_calculation_l538_53814


namespace NUMINAMATH_CALUDE_crushing_load_calculation_l538_53851

theorem crushing_load_calculation (T H D : ℝ) (hT : T = 5) (hH : H = 15) (hD : D = 10) :
  let L := (30 * T^3) / (H * D)
  L = 25 := by
sorry

end NUMINAMATH_CALUDE_crushing_load_calculation_l538_53851


namespace NUMINAMATH_CALUDE_implication_equivalence_l538_53867

theorem implication_equivalence (R S : Prop) :
  (R → S) ↔ (¬S → ¬R) :=
by sorry

end NUMINAMATH_CALUDE_implication_equivalence_l538_53867


namespace NUMINAMATH_CALUDE_g_x_squared_properties_l538_53838

noncomputable def f (x : ℝ) : ℝ := (1/2)^x

def symmetric_wrt_y_eq_x (f g : ℝ → ℝ) : Prop :=
  ∀ x y, f x = y ↔ g y = x

theorem g_x_squared_properties
  (g : ℝ → ℝ)
  (h_sym : symmetric_wrt_y_eq_x f g) :
  (∀ x, g (x^2) = g ((-x)^2)) ∧
  (∀ x y, x < y → x < 0 → y < 0 → g (x^2) < g (y^2)) :=
sorry

end NUMINAMATH_CALUDE_g_x_squared_properties_l538_53838


namespace NUMINAMATH_CALUDE_tens_digit_of_3_to_2013_l538_53832

theorem tens_digit_of_3_to_2013 : ∃ n : ℕ, 3^2013 ≡ 40 + n [ZMOD 100] ∧ n < 10 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_3_to_2013_l538_53832


namespace NUMINAMATH_CALUDE_min_value_theorem_l538_53835

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 4 / b = 1) :
  ∃ (m : ℝ), m = 18 ∧ ∀ (x : ℝ), 2 / a + 2 * b ≥ x → m ≤ x :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l538_53835
