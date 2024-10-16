import Mathlib

namespace NUMINAMATH_CALUDE_nearest_multiple_of_21_to_2304_l2033_203313

theorem nearest_multiple_of_21_to_2304 :
  ∀ n : ℤ, n ≠ 2304 → 21 ∣ n → |n - 2304| ≥ |2310 - 2304| :=
by sorry

end NUMINAMATH_CALUDE_nearest_multiple_of_21_to_2304_l2033_203313


namespace NUMINAMATH_CALUDE_intersection_nonempty_condition_l2033_203360

theorem intersection_nonempty_condition (m n : ℝ) :
  let A := {x : ℝ | m - 1 < x ∧ x < m + 1}
  let B := {x : ℝ | 3 - n < x ∧ x < 4 - n}
  (∃ x, x ∈ A ∩ B) ↔ 2 < m + n ∧ m + n < 5 :=
by sorry

end NUMINAMATH_CALUDE_intersection_nonempty_condition_l2033_203360


namespace NUMINAMATH_CALUDE_largest_zip_code_l2033_203335

def phone_number : List Nat := [4, 6, 5, 3, 2, 7, 1]

def is_valid_zip_code (zip : List Nat) : Prop :=
  zip.length = 4 ∧ 
  zip.toFinset.card = 4 ∧
  zip.sum = phone_number.sum

def zip_code_value (zip : List Nat) : Nat :=
  zip.foldl (fun acc d => acc * 10 + d) 0

theorem largest_zip_code :
  ∀ zip : List Nat, is_valid_zip_code zip →
  zip_code_value zip ≤ 9865 :=
sorry

end NUMINAMATH_CALUDE_largest_zip_code_l2033_203335


namespace NUMINAMATH_CALUDE_whitewashing_cost_l2033_203321

-- Define the room dimensions
def room_length : ℝ := 25
def room_width : ℝ := 15
def room_height : ℝ := 12

-- Define the door dimensions
def door_height : ℝ := 6
def door_width : ℝ := 3

-- Define the window dimensions
def window_height : ℝ := 4
def window_width : ℝ := 3
def num_windows : ℕ := 3

-- Define the cost per square foot
def cost_per_sqft : ℝ := 8

-- Theorem statement
theorem whitewashing_cost :
  let total_wall_area := 2 * (room_length + room_width) * room_height
  let door_area := door_height * door_width
  let window_area := window_height * window_width * num_windows
  let effective_area := total_wall_area - door_area - window_area
  effective_area * cost_per_sqft = 7248 := by
sorry


end NUMINAMATH_CALUDE_whitewashing_cost_l2033_203321


namespace NUMINAMATH_CALUDE_x_times_one_minus_f_equals_one_l2033_203303

/-- Given x = (3 + 2√2)^1000, n = ⌊x⌋, and f = x - n, prove that x(1 - f) = 1 -/
theorem x_times_one_minus_f_equals_one :
  let x : ℝ := (3 + 2 * Real.sqrt 2) ^ 1000
  let n : ℤ := ⌊x⌋
  let f : ℝ := x - n
  x * (1 - f) = 1 := by
sorry

end NUMINAMATH_CALUDE_x_times_one_minus_f_equals_one_l2033_203303


namespace NUMINAMATH_CALUDE_multiply_121_54_l2033_203383

theorem multiply_121_54 : 121 * 54 = 6534 := by sorry

end NUMINAMATH_CALUDE_multiply_121_54_l2033_203383


namespace NUMINAMATH_CALUDE_curve_symmetry_condition_l2033_203328

/-- Given a curve y = x + p/x where p ≠ 0, this theorem states that the condition for two distinct
points on the curve to be symmetric with respect to the line y = x is satisfied if and only if p < 0 -/
theorem curve_symmetry_condition (p : ℝ) (hp : p ≠ 0) :
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
   x₁ + p / x₁ = x₂ + p / x₂ ∧
   x₁ + p / x₁ + x₂ + p / x₂ = x₁ + x₂) ↔ 
  p < 0 := by
  sorry

end NUMINAMATH_CALUDE_curve_symmetry_condition_l2033_203328


namespace NUMINAMATH_CALUDE_gcd_18_30_l2033_203334

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_18_30_l2033_203334


namespace NUMINAMATH_CALUDE_girl_speed_l2033_203305

/-- Given a girl traveling a distance of 96 meters in 16 seconds,
    prove that her speed is 6 meters per second. -/
theorem girl_speed (distance : ℝ) (time : ℝ) (speed : ℝ) 
    (h1 : distance = 96) 
    (h2 : time = 16) 
    (h3 : speed = distance / time) : 
  speed = 6 := by
  sorry

end NUMINAMATH_CALUDE_girl_speed_l2033_203305


namespace NUMINAMATH_CALUDE_cube_surface_area_l2033_203393

/-- The surface area of a cube with edge length 4a is 96a² -/
theorem cube_surface_area (a : ℝ) : 
  let edge_length : ℝ := 4 * a
  let surface_area : ℝ := 6 * (edge_length ^ 2)
  surface_area = 96 * (a ^ 2) := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_l2033_203393


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_l2033_203345

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + 4*y + x*y = 1) : 
  ∃ (m : ℝ), m = 2*Real.sqrt 6 - 4 ∧ x + 2*y ≥ m ∧ ∀ z, (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 2*a + 4*b + a*b = 1 ∧ z = a + 2*b) → z ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_l2033_203345


namespace NUMINAMATH_CALUDE_least_number_added_for_divisibility_l2033_203391

theorem least_number_added_for_divisibility (x : ℕ) : 
  (∀ y : ℕ, y < x → ¬(23 ∣ (1054 + y))) ∧ (23 ∣ (1054 + x)) → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_least_number_added_for_divisibility_l2033_203391


namespace NUMINAMATH_CALUDE_arithmetic_mean_reciprocals_first_four_primes_l2033_203306

theorem arithmetic_mean_reciprocals_first_four_primes :
  let primes : List ℕ := [2, 3, 5, 7]
  let reciprocals := primes.map (λ x => (1 : ℚ) / x)
  let sum := reciprocals.sum
  let mean := sum / 4
  mean = 247 / 840 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_reciprocals_first_four_primes_l2033_203306


namespace NUMINAMATH_CALUDE_cake_division_l2033_203347

theorem cake_division (K : ℕ) (h_K : K = 1997) : 
  ∃ N : ℕ, 
    (N > 0) ∧ 
    (K ∣ N) ∧ 
    (K ∣ N^3) ∧ 
    (K ∣ 6*N^2) ∧ 
    (∀ M : ℕ, M < N → ¬(K ∣ M) ∨ ¬(K ∣ M^3) ∨ ¬(K ∣ 6*M^2)) :=
by sorry

end NUMINAMATH_CALUDE_cake_division_l2033_203347


namespace NUMINAMATH_CALUDE_product_equals_99999919_l2033_203338

theorem product_equals_99999919 : 103 * 97 * 10009 = 99999919 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_99999919_l2033_203338


namespace NUMINAMATH_CALUDE_roses_kept_l2033_203300

theorem roses_kept (total : ℕ) (to_mother : ℕ) (to_grandmother : ℕ) (to_sister : ℕ) 
  (h1 : total = 20)
  (h2 : to_mother = 6)
  (h3 : to_grandmother = 9)
  (h4 : to_sister = 4) : 
  total - (to_mother + to_grandmother + to_sister) = 1 := by
  sorry

end NUMINAMATH_CALUDE_roses_kept_l2033_203300


namespace NUMINAMATH_CALUDE_angus_has_55_tokens_l2033_203367

/-- The number of tokens Angus has -/
def angus_tokens (elsa_tokens : ℕ) (token_value : ℕ) (value_difference : ℕ) : ℕ :=
  elsa_tokens - (value_difference / token_value)

/-- Theorem stating that Angus has 55 tokens -/
theorem angus_has_55_tokens (elsa_tokens : ℕ) (token_value : ℕ) (value_difference : ℕ)
  (h1 : elsa_tokens = 60)
  (h2 : token_value = 4)
  (h3 : value_difference = 20) :
  angus_tokens elsa_tokens token_value value_difference = 55 := by
  sorry

end NUMINAMATH_CALUDE_angus_has_55_tokens_l2033_203367


namespace NUMINAMATH_CALUDE_correct_percentage_l2033_203314

theorem correct_percentage (x : ℕ) : 
  let total := 6 * x
  let missed := 2 * x
  let correct := total - missed
  (correct : ℚ) / total * 100 = 200 / 3 := by sorry

end NUMINAMATH_CALUDE_correct_percentage_l2033_203314


namespace NUMINAMATH_CALUDE_ones_digit_of_large_power_l2033_203363

theorem ones_digit_of_large_power : ∃ n : ℕ, n > 0 ∧ 17^(17*(5^5)) ≡ 7 [ZMOD 10] := by
  sorry

end NUMINAMATH_CALUDE_ones_digit_of_large_power_l2033_203363


namespace NUMINAMATH_CALUDE_coloring_exists_l2033_203357

def M : Set ℕ := {n | 1 ≤ n ∧ n ≤ 2010}

theorem coloring_exists :
  ∃ (f : ℕ → Fin 5),
    ∀ (a d : ℕ),
      a ∈ M →
      d > 0 →
      (∀ k, k ∈ Finset.range 9 → (a + k * d) ∈ M) →
      ∃ (i j : Fin 9), i ≠ j ∧ f (a + i * d) ≠ f (a + j * d) :=
by sorry

end NUMINAMATH_CALUDE_coloring_exists_l2033_203357


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l2033_203370

/-- A rhombus with given diagonal lengths has a specific perimeter. -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) :
  let side := Real.sqrt ((d1/2)^2 + (d2/2)^2)
  4 * side = 52 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l2033_203370


namespace NUMINAMATH_CALUDE_negative_fraction_comparison_l2033_203302

theorem negative_fraction_comparison :
  -3/4 > -4/5 :=
by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_comparison_l2033_203302


namespace NUMINAMATH_CALUDE_baker_remaining_cakes_l2033_203317

/-- Given a baker who made 167 cakes and sold 108 cakes, prove that the number of cakes remaining is 59. -/
theorem baker_remaining_cakes (cakes_made : ℕ) (cakes_sold : ℕ) 
  (h1 : cakes_made = 167) (h2 : cakes_sold = 108) : 
  cakes_made - cakes_sold = 59 := by
  sorry

#check baker_remaining_cakes

end NUMINAMATH_CALUDE_baker_remaining_cakes_l2033_203317


namespace NUMINAMATH_CALUDE_card_combination_problem_l2033_203368

theorem card_combination_problem : Nat.choose 60 8 = 7580800000 := by
  sorry

end NUMINAMATH_CALUDE_card_combination_problem_l2033_203368


namespace NUMINAMATH_CALUDE_dragon_jewels_l2033_203387

theorem dragon_jewels (D : ℕ) : 
  6 = D / 3 →  -- The new jewels (6) are one-third of the original count
  21 = D - 3 + 6 -- The final count is the original count minus 3 (stolen) plus 6 (taken from king)
  := by sorry

end NUMINAMATH_CALUDE_dragon_jewels_l2033_203387


namespace NUMINAMATH_CALUDE_trapezoid_segment_length_l2033_203348

/-- Represents a line segment with a length -/
structure Segment where
  length : ℝ

/-- Represents a trapezoid ABCD with AB parallel to CD -/
structure Trapezoid where
  AB : Segment
  CD : Segment
  EF : Segment
  AB_parallel_CD : True  -- Represents AB ∥ CD
  EF_parallel_AB : True  -- Represents EF ∥ AB
  EF_parallel_CD : True  -- Represents EF ∥ CD

/-- Theorem: In a trapezoid ABCD with EF parallel to both AB and CD,
    if AB = 200 cm and EF = 50 cm, then CD = 50 cm -/
theorem trapezoid_segment_length (t : Trapezoid)
    (h1 : t.AB.length = 200)
    (h2 : t.EF.length = 50) :
    t.CD.length = 50 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_segment_length_l2033_203348


namespace NUMINAMATH_CALUDE_birdseed_mix_theorem_l2033_203353

/-- The percentage of millet in Brand A birdseed -/
def brand_a_millet : ℝ := sorry

/-- The percentage of millet in Brand B birdseed -/
def brand_b_millet : ℝ := 0.65

/-- The percentage of Brand A in the mix -/
def mix_brand_a : ℝ := 0.60

/-- The percentage of Brand B in the mix -/
def mix_brand_b : ℝ := 0.40

/-- The percentage of millet in the final mix -/
def mix_millet : ℝ := 0.50

theorem birdseed_mix_theorem :
  mix_brand_a * brand_a_millet + mix_brand_b * brand_b_millet = mix_millet ∧
  brand_a_millet = 0.40 :=
sorry

end NUMINAMATH_CALUDE_birdseed_mix_theorem_l2033_203353


namespace NUMINAMATH_CALUDE_solve_a_and_b_l2033_203343

theorem solve_a_and_b : ∃ (a b : ℝ), 
  (b^2 - 2*b = 24) ∧ 
  (4*(1:ℝ)^2 + a = 2) ∧ 
  (4*b^2 - 2*b = 72) ∧ 
  (a = -2) ∧ 
  (b = -4) := by
  sorry

end NUMINAMATH_CALUDE_solve_a_and_b_l2033_203343


namespace NUMINAMATH_CALUDE_victoria_wheat_flour_packets_l2033_203362

/-- Calculates the number of wheat flour packets bought given the initial amount,
    costs of items, and remaining balance. -/
def wheat_flour_packets (initial_amount : ℕ) (rice_cost : ℕ) (rice_packets : ℕ) 
                        (soda_cost : ℕ) (wheat_flour_cost : ℕ) (remaining_balance : ℕ) : ℕ :=
  let total_spent := initial_amount - remaining_balance
  let rice_soda_cost := rice_cost * rice_packets + soda_cost
  let wheat_flour_total := total_spent - rice_soda_cost
  wheat_flour_total / wheat_flour_cost

/-- Theorem stating that Victoria bought 3 packets of wheat flour -/
theorem victoria_wheat_flour_packets : 
  wheat_flour_packets 500 20 2 150 25 235 = 3 := by
  sorry

end NUMINAMATH_CALUDE_victoria_wheat_flour_packets_l2033_203362


namespace NUMINAMATH_CALUDE_bookcase_length_inches_l2033_203330

/-- Conversion factor from feet to inches -/
def inches_per_foot : ℕ := 12

/-- Length of the bookcase in feet -/
def bookcase_length_feet : ℕ := 4

/-- Theorem stating that a 4-foot bookcase is 48 inches long -/
theorem bookcase_length_inches : 
  bookcase_length_feet * inches_per_foot = 48 := by
  sorry

end NUMINAMATH_CALUDE_bookcase_length_inches_l2033_203330


namespace NUMINAMATH_CALUDE_original_not_imply_converse_converse_implies_negation_l2033_203342

-- Define a proposition P and Q
variable (P Q : Prop)

-- Statement 1: The truth of an original statement does not necessarily imply the truth of its converse
theorem original_not_imply_converse : ∃ P Q, (P → Q) ∧ ¬(Q → P) := by sorry

-- Statement 2: If the converse of a statement is true, then its negation is also true
theorem converse_implies_negation : ∀ P Q, (Q → P) → (¬P → ¬Q) := by sorry

end NUMINAMATH_CALUDE_original_not_imply_converse_converse_implies_negation_l2033_203342


namespace NUMINAMATH_CALUDE_calculate_interest_rate_l2033_203364

/-- Given a sum of money invested at simple interest, this theorem proves
    that if the interest earned is a certain amount more than what would
    be earned at a reference rate, then the actual interest rate can be
    calculated. -/
theorem calculate_interest_rate 
  (principal : ℝ) 
  (time : ℝ) 
  (reference_rate : ℝ) 
  (interest_difference : ℝ) 
  (h1 : principal = 4200)
  (h2 : time = 2)
  (h3 : reference_rate = 0.12)
  (h4 : interest_difference = 504)
  : (principal * time * reference_rate + interest_difference) / (principal * time) = 0.18 := by
  sorry

end NUMINAMATH_CALUDE_calculate_interest_rate_l2033_203364


namespace NUMINAMATH_CALUDE_money_sharing_l2033_203301

theorem money_sharing (amanda_share : ℕ) (total : ℕ) : 
  amanda_share = 30 →
  3 * total = 16 * amanda_share →
  total = 160 := by sorry

end NUMINAMATH_CALUDE_money_sharing_l2033_203301


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2033_203389

theorem absolute_value_inequality (a b : ℝ) : 
  (1 / |a| < 1 / |b|) → |a| > |b| := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2033_203389


namespace NUMINAMATH_CALUDE_intersection_sum_l2033_203316

-- Define the two parabolas
def parabola1 (x y : ℝ) : Prop := y = (x - 2)^2
def parabola2 (x y : ℝ) : Prop := x + 6 = (y + 1)^2

-- Define the intersection points
def intersection_points : Prop := ∃ x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ,
  (parabola1 x₁ y₁ ∧ parabola2 x₁ y₁) ∧
  (parabola1 x₂ y₂ ∧ parabola2 x₂ y₂) ∧
  (parabola1 x₃ y₃ ∧ parabola2 x₃ y₃) ∧
  (parabola1 x₄ y₄ ∧ parabola2 x₄ y₄)

-- Theorem statement
theorem intersection_sum : intersection_points →
  ∃ x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ,
  (parabola1 x₁ y₁ ∧ parabola2 x₁ y₁) ∧
  (parabola1 x₂ y₂ ∧ parabola2 x₂ y₂) ∧
  (parabola1 x₃ y₃ ∧ parabola2 x₃ y₃) ∧
  (parabola1 x₄ y₄ ∧ parabola2 x₄ y₄) ∧
  x₁ + x₂ + x₃ + x₄ + y₁ + y₂ + y₃ + y₄ = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_l2033_203316


namespace NUMINAMATH_CALUDE_different_color_probability_l2033_203307

/-- The probability of drawing two balls of different colors from a box -/
theorem different_color_probability (total_balls : ℕ) (white_balls : ℕ) (black_balls : ℕ) :
  total_balls = white_balls + black_balls →
  white_balls = 3 →
  black_balls = 2 →
  (white_balls * black_balls : ℚ) / ((total_balls * (total_balls - 1)) / 2 : ℚ) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_different_color_probability_l2033_203307


namespace NUMINAMATH_CALUDE_euler_totient_properties_l2033_203351

/-- Euler's totient function -/
def phi (n : ℕ) : ℕ := sorry

/-- Definition: p is prime -/
def is_prime (p : ℕ) : Prop := sorry

theorem euler_totient_properties (p : ℕ) (α : ℕ) (h : is_prime p) (h' : α > 0) :
  (phi 17 = 16) ∧
  (phi p = p - 1) ∧
  (phi (p^2) = p * (p - 1)) ∧
  (phi (p^α) = p^(α-1) * (p - 1)) :=
sorry

end NUMINAMATH_CALUDE_euler_totient_properties_l2033_203351


namespace NUMINAMATH_CALUDE_jenny_project_hours_l2033_203369

/-- The total hours Jenny has to work on her school project -/
def total_project_hours (research_hours proposal_hours report_hours : ℕ) : ℕ :=
  research_hours + proposal_hours + report_hours

/-- Theorem stating that Jenny's total project hours is 20 -/
theorem jenny_project_hours :
  total_project_hours 10 2 8 = 20 := by
  sorry

end NUMINAMATH_CALUDE_jenny_project_hours_l2033_203369


namespace NUMINAMATH_CALUDE_hyperbola_circle_tangency_l2033_203325

/-- Given a hyperbola and a circle satisfying certain conditions, prove the values of a² and b² -/
theorem hyperbola_circle_tangency (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 →  -- Hyperbola equation
    ∃ t : ℝ, (a * t)^2 + (b * t)^2 = (x - 3)^2 + y^2) →  -- Asymptotes touch the circle
  (a^2 + b^2 = 9) →  -- Right focus coincides with circle center
  (a^2 = 5 ∧ b^2 = 4) := by sorry

end NUMINAMATH_CALUDE_hyperbola_circle_tangency_l2033_203325


namespace NUMINAMATH_CALUDE_hockey_goals_difference_l2033_203376

theorem hockey_goals_difference (layla_goals kristin_goals : ℕ) : 
  layla_goals = 104 →
  kristin_goals < layla_goals →
  (layla_goals + kristin_goals) / 2 = 92 →
  layla_goals - kristin_goals = 24 := by
sorry

end NUMINAMATH_CALUDE_hockey_goals_difference_l2033_203376


namespace NUMINAMATH_CALUDE_duanes_initial_pages_l2033_203356

theorem duanes_initial_pages (lana_initial : ℕ) (lana_final : ℕ) (duane_initial : ℕ) : 
  lana_initial = 8 → 
  lana_final = 29 → 
  lana_final = lana_initial + duane_initial / 2 →
  duane_initial = 42 := by
sorry

end NUMINAMATH_CALUDE_duanes_initial_pages_l2033_203356


namespace NUMINAMATH_CALUDE_unique_sum_of_equation_l2033_203319

theorem unique_sum_of_equation (x y : ℤ) :
  (1 / x + 1 / y) * (1 / x^2 + 1 / y^2) = -2/3 * (1 / x^4 - 1 / y^4) →
  ∃! s : ℤ, s = x + y :=
by sorry

end NUMINAMATH_CALUDE_unique_sum_of_equation_l2033_203319


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_property_l2033_203326

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ (d : ℚ), ∀ n, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def SumArithmeticSequence (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2

theorem arithmetic_sequence_ratio_property (a : ℕ → ℚ) 
    (h_arith : ArithmeticSequence a) (h_ratio : a 7 / a 4 = 7 / 13) :
    SumArithmeticSequence a 13 / SumArithmeticSequence a 7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_property_l2033_203326


namespace NUMINAMATH_CALUDE_divisibility_of_polynomial_l2033_203350

theorem divisibility_of_polynomial (x : ℕ) (h_prime : Nat.Prime x) (h_gt3 : x > 3) :
  (∃ n : ℤ, x = 3 * n + 1 ∧ (x^6 - x^3 - x^2 + x) % 12 = 0) ∨
  (∃ n : ℤ, x = 3 * n - 1 ∧ (x^6 - x^3 - x^2 + x) % 36 = 0) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_of_polynomial_l2033_203350


namespace NUMINAMATH_CALUDE_max_value_trig_expression_l2033_203344

theorem max_value_trig_expression :
  ∀ x : ℝ, 2 * Real.cos x + 3 * Real.sin x ≤ Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_max_value_trig_expression_l2033_203344


namespace NUMINAMATH_CALUDE_cube_edge_length_l2033_203386

theorem cube_edge_length (volume : ℝ) (edge_length : ℝ) :
  volume = 2744 ∧ volume = edge_length ^ 3 → edge_length = 14 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_length_l2033_203386


namespace NUMINAMATH_CALUDE_dodecahedron_edge_probability_l2033_203318

/-- A regular dodecahedron -/
structure RegularDodecahedron :=
  (vertices : ℕ)
  (edges_per_vertex : ℕ)
  (h_vertices : vertices = 20)
  (h_edges_per_vertex : edges_per_vertex = 3)

/-- The probability of two randomly chosen vertices being connected by an edge -/
def edge_probability (d : RegularDodecahedron) : ℚ :=
  3 / 19

theorem dodecahedron_edge_probability (d : RegularDodecahedron) :
  edge_probability d = 3 / 19 :=
by sorry

end NUMINAMATH_CALUDE_dodecahedron_edge_probability_l2033_203318


namespace NUMINAMATH_CALUDE_bryans_score_l2033_203380

/-- Represents the math exam scores for Bryan, Jen, and Sammy -/
structure ExamScores where
  bryan : ℕ
  jen : ℕ
  sammy : ℕ

/-- The total points possible on the exam -/
def totalPoints : ℕ := 35

/-- Defines the relationship between the scores based on the given conditions -/
def validScores (scores : ExamScores) : Prop :=
  scores.jen = scores.bryan + 10 ∧
  scores.sammy = scores.jen - 2 ∧
  scores.sammy = totalPoints - 7

/-- Theorem stating Bryan's score on the exam -/
theorem bryans_score (scores : ExamScores) (h : validScores scores) : scores.bryan = 20 := by
  sorry

end NUMINAMATH_CALUDE_bryans_score_l2033_203380


namespace NUMINAMATH_CALUDE_odd_function_property_l2033_203399

-- Define the domain D
def D : Set ℝ := {x : ℝ | x ≠ 0}

-- Define the properties of the function f
def is_odd_function_on_D (f : ℝ → ℝ) : Prop :=
  ∀ x ∈ D, f (-x) = -f x

-- State the theorem
theorem odd_function_property
  (f : ℝ → ℝ)
  (h_odd : is_odd_function_on_D f)
  (h_pos : ∀ x > 0, f x = x^2 - x) :
  ∀ x < 0, f x = -x^2 - x :=
sorry

end NUMINAMATH_CALUDE_odd_function_property_l2033_203399


namespace NUMINAMATH_CALUDE_water_bill_calculation_l2033_203374

/-- Water bill calculation for a household --/
theorem water_bill_calculation 
  (a : ℝ) -- Base rate for water usage up to 20 cubic meters
  (usage : ℝ) -- Total water usage
  (h1 : usage = 25) -- The household used 25 cubic meters
  (h2 : usage > 20) -- Usage exceeds 20 cubic meters
  : 
  (min usage 20) * a + (usage - 20) * (a + 3) = 25 * a + 15 :=
by sorry

end NUMINAMATH_CALUDE_water_bill_calculation_l2033_203374


namespace NUMINAMATH_CALUDE_regression_maximum_fitting_l2033_203379

/-- Represents a linear regression model --/
structure LinearRegression where
  slope : ℝ
  intercept : ℝ

/-- Represents the true relationship between x and y --/
def true_relationship : ℝ → ℝ := sorry

/-- Measures the degree of fitting between the regression model and the true relationship --/
def fitting_degree (model : LinearRegression) : ℝ := sorry

/-- The regression equation represents the maximum degree of fitting --/
theorem regression_maximum_fitting (data : List (ℝ × ℝ)) :
  ∃ (model : LinearRegression),
    ∀ (other_model : LinearRegression),
      fitting_degree model ≥ fitting_degree other_model := by
  sorry

end NUMINAMATH_CALUDE_regression_maximum_fitting_l2033_203379


namespace NUMINAMATH_CALUDE_exactly_two_valid_numbers_l2033_203333

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def valid_number (n : ℕ) : Prop :=
  (n ≥ 1000 ∧ n ≤ 9999) ∧
  is_perfect_square (n / 1000 + (n / 100 % 10) + (n / 10 % 10) + (n % 10)) ∧
  is_perfect_square ((n / 10 % 10) + (n % 10)) ∧
  is_perfect_square ((n / 10 % 10) - (n % 10)) ∧
  is_perfect_square (n % 10) ∧
  is_perfect_square ((n / 100) % 100) ∧
  (n / 1000 + (n / 100 % 10) + (n / 10 % 10) + (n % 10) > 0) ∧
  ((n / 10 % 10) + (n % 10) > 0) ∧
  ((n / 10 % 10) - (n % 10) > 0) ∧
  (n % 10 > 0)

theorem exactly_two_valid_numbers :
  ∃! (s : Finset ℕ), s.card = 2 ∧ ∀ n ∈ s, valid_number n :=
sorry

end NUMINAMATH_CALUDE_exactly_two_valid_numbers_l2033_203333


namespace NUMINAMATH_CALUDE_partner_investment_time_l2033_203378

/-- Given two partners P and Q with investments and profits, prove Q's investment time -/
theorem partner_investment_time 
  (investment_ratio : ℚ) -- Ratio of P's investment to Q's investment
  (profit_ratio : ℚ) -- Ratio of P's profit to Q's profit
  (p_time : ℕ) -- Time P invested in months
  (h1 : investment_ratio = 7 / 5)
  (h2 : profit_ratio = 7 / 14)
  (h3 : p_time = 5) :
  ∃ (q_time : ℕ), q_time = 14 := by
  sorry


end NUMINAMATH_CALUDE_partner_investment_time_l2033_203378


namespace NUMINAMATH_CALUDE_expand_polynomial_l2033_203309

theorem expand_polynomial (x : ℝ) : 
  (x - 2) * (x + 2) * (x^3 + 3*x + 1) = x^5 - x^3 + x^2 - 12*x - 4 := by
  sorry

end NUMINAMATH_CALUDE_expand_polynomial_l2033_203309


namespace NUMINAMATH_CALUDE_divisibility_property_l2033_203304

theorem divisibility_property (a b c d u : ℤ) 
  (h1 : u ∣ a * c) 
  (h2 : u ∣ b * c + a * d) 
  (h3 : u ∣ b * d) : 
  (u ∣ b * c) ∧ (u ∣ a * d) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l2033_203304


namespace NUMINAMATH_CALUDE_odd_function_unique_solution_l2033_203377

def f (a b c : ℤ) (x : ℚ) : ℚ := (a * x^2 + 1) / (b * x + c)

theorem odd_function_unique_solution :
  ∀ a b c : ℤ,
  (∀ x : ℚ, f a b c (-x) = -(f a b c x)) →
  f a b c 1 = 2 →
  f a b c 2 < 3 →
  a = 1 ∧ b = 1 ∧ c = 0 := by
sorry

end NUMINAMATH_CALUDE_odd_function_unique_solution_l2033_203377


namespace NUMINAMATH_CALUDE_kelly_games_l2033_203390

theorem kelly_games (initial_games given_away left : ℕ) : 
  given_away = 99 → left = 22 → initial_games = given_away + left :=
by sorry

end NUMINAMATH_CALUDE_kelly_games_l2033_203390


namespace NUMINAMATH_CALUDE_point_product_y_coordinates_l2033_203331

theorem point_product_y_coordinates : 
  ∀ y₁ y₂ : ℝ, 
  (3 - 1)^2 + (-1 - y₁)^2 = 10^2 →
  (3 - 1)^2 + (-1 - y₂)^2 = 10^2 →
  y₁ * y₂ = -95 := by
sorry

end NUMINAMATH_CALUDE_point_product_y_coordinates_l2033_203331


namespace NUMINAMATH_CALUDE_cookie_radius_l2033_203371

theorem cookie_radius (x y : ℝ) :
  (x^2 + y^2 + 36 = 6*x + 9*y) →
  ∃ (center_x center_y : ℝ), 
    (x - center_x)^2 + (y - center_y)^2 = (3*Real.sqrt 5 / 2)^2 :=
by sorry

end NUMINAMATH_CALUDE_cookie_radius_l2033_203371


namespace NUMINAMATH_CALUDE_cubic_root_sum_l2033_203373

/-- Given a cubic equation ax³ + bx² + cx + d = 0 with a ≠ 0,
    if 3 and -2 are roots of the equation, then (b+c)/a = -7 -/
theorem cubic_root_sum (a b c d : ℝ) (ha : a ≠ 0) 
  (h1 : a * 3^3 + b * 3^2 + c * 3 + d = 0)
  (h2 : a * (-2)^3 + b * (-2)^2 + c * (-2) + d = 0) :
  (b + c) / a = -7 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l2033_203373


namespace NUMINAMATH_CALUDE_fathers_age_l2033_203308

theorem fathers_age (son_age father_age : ℕ) : 
  father_age = 3 * son_age + 3 →
  father_age + 3 = 2 * (son_age + 3) + 8 →
  father_age = 27 := by
sorry

end NUMINAMATH_CALUDE_fathers_age_l2033_203308


namespace NUMINAMATH_CALUDE_book_pages_from_digits_l2033_203366

theorem book_pages_from_digits (total_digits : ℕ) : total_digits = 792 → ∃ (pages : ℕ), pages = 300 ∧ 
  (pages ≤ 9 → total_digits = pages) ∧
  (9 < pages ∧ pages ≤ 99 → total_digits = 9 + 2 * (pages - 9)) ∧
  (99 < pages → total_digits = 189 + 3 * (pages - 99)) :=
by
  sorry

end NUMINAMATH_CALUDE_book_pages_from_digits_l2033_203366


namespace NUMINAMATH_CALUDE_parking_problem_l2033_203358

/-- Represents the number of parking spaces -/
def total_spaces : ℕ := 7

/-- Represents the number of cars -/
def num_cars : ℕ := 3

/-- Represents the number of consecutive empty spaces -/
def empty_spaces : ℕ := 4

/-- Represents the total number of units to arrange (cars + empty space block) -/
def total_units : ℕ := num_cars + 1

/-- The number of different parking arrangements -/
def parking_arrangements : ℕ := Nat.factorial total_units

theorem parking_problem :
  parking_arrangements = 24 :=
sorry

end NUMINAMATH_CALUDE_parking_problem_l2033_203358


namespace NUMINAMATH_CALUDE_divide_number_with_percentage_condition_l2033_203349

theorem divide_number_with_percentage_condition : 
  ∃ (x : ℝ), 
    x + (80 - x) = 80 ∧ 
    0.3 * x = 0.2 * (80 - x) + 10 ∧ 
    min x (80 - x) = 28 := by
  sorry

end NUMINAMATH_CALUDE_divide_number_with_percentage_condition_l2033_203349


namespace NUMINAMATH_CALUDE_sum_of_first_50_even_integers_l2033_203310

theorem sum_of_first_50_even_integers (sum_odd : ℕ) : 
  sum_odd = 50^2 → 
  (Finset.sum (Finset.range 50) (λ i => 2*i + 2) = sum_odd + 50) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_first_50_even_integers_l2033_203310


namespace NUMINAMATH_CALUDE_y₁_not_in_third_quadrant_l2033_203382

-- Define the linear functions
def y₁ (x : ℝ) (b : ℝ) : ℝ := -x + b
def y₂ (x : ℝ) : ℝ := -x

-- State the theorem
theorem y₁_not_in_third_quadrant :
  ∃ b : ℝ, (∀ x : ℝ, y₁ x b = y₂ x + 2) →
  ∀ x y : ℝ, y = y₁ x b → (x < 0 ∧ y < 0 → False) := by
  sorry

end NUMINAMATH_CALUDE_y₁_not_in_third_quadrant_l2033_203382


namespace NUMINAMATH_CALUDE_weight_loss_challenge_l2033_203394

/-- 
Given an initial weight W, a weight loss percentage, and a clothing weight percentage,
calculates the final measured weight loss percentage.
-/
def measured_weight_loss_percentage (initial_weight_loss : Real) (clothing_weight_percent : Real) : Real :=
  let remaining_weight_percent := 1 - initial_weight_loss
  let final_weight_percent := remaining_weight_percent * (1 + clothing_weight_percent)
  (1 - final_weight_percent) * 100

/-- 
Proves that given an initial weight loss of 15% and clothes that add 2% to the final weight,
the measured weight loss percentage at the final weigh-in is 13.3%.
-/
theorem weight_loss_challenge (ε : Real) :
  ∃ δ > 0, ∀ x, |x - 0.133| < δ → |measured_weight_loss_percentage 0.15 0.02 - x| < ε :=
sorry

end NUMINAMATH_CALUDE_weight_loss_challenge_l2033_203394


namespace NUMINAMATH_CALUDE_simplify_expression_l2033_203323

theorem simplify_expression (a : ℝ) : 2*a + 1 - (1 - a) = 3*a := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2033_203323


namespace NUMINAMATH_CALUDE_repunit_existence_l2033_203312

theorem repunit_existence (p : Nat) (h_prime : Nat.Prime p) (h_p_gt_11 : p > 11) :
  ∃ k : Nat, ∃ n : Nat, p * k = (10^n - 1) / 9 := by
  sorry

end NUMINAMATH_CALUDE_repunit_existence_l2033_203312


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l2033_203329

theorem simplify_sqrt_expression (x : ℝ) (hx : x ≠ 0) :
  Real.sqrt (1 + ((x^6 - 1) / (3 * x^3))^2) = (Real.sqrt (x^12 + 7 * x^6 + 1)) / (3 * x^3) :=
sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l2033_203329


namespace NUMINAMATH_CALUDE_circle_equation_from_diameter_endpoints_l2033_203395

theorem circle_equation_from_diameter_endpoints (x y : ℝ) :
  let p₁ : ℝ × ℝ := (0, 0)
  let p₂ : ℝ × ℝ := (6, 8)
  let center : ℝ × ℝ := ((p₁.1 + p₂.1) / 2, (p₁.2 + p₂.2) / 2)
  let radius : ℝ := Real.sqrt ((p₂.1 - p₁.1)^2 + (p₂.2 - p₁.2)^2) / 2
  (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by
  sorry

end NUMINAMATH_CALUDE_circle_equation_from_diameter_endpoints_l2033_203395


namespace NUMINAMATH_CALUDE_hiking_team_participants_l2033_203384

theorem hiking_team_participants (total_gloves : ℕ) (gloves_per_participant : ℕ) : 
  total_gloves = 164 → gloves_per_participant = 2 → total_gloves / gloves_per_participant = 82 := by
  sorry

end NUMINAMATH_CALUDE_hiking_team_participants_l2033_203384


namespace NUMINAMATH_CALUDE_constant_solution_implies_product_l2033_203322

/-- 
Given constants a and b, if the equation (2kx+a)/3 = 2 + (x-bk)/6 
always has a solution of x = 1 for any k, then ab = -26
-/
theorem constant_solution_implies_product (a b : ℚ) : 
  (∀ k : ℚ, ∃ x : ℚ, x = 1 ∧ (2*k*x + a) / 3 = 2 + (x - b*k) / 6) → 
  a * b = -26 := by
sorry

end NUMINAMATH_CALUDE_constant_solution_implies_product_l2033_203322


namespace NUMINAMATH_CALUDE_debt_average_payment_l2033_203392

/-- Prove that the average payment for a debt with specific payment structure is $442.50 -/
theorem debt_average_payment (n : ℕ) (first_payment second_payment : ℚ) : 
  n = 40 →
  first_payment = 410 →
  second_payment = first_payment + 65 →
  (n / 2 * first_payment + n / 2 * second_payment) / n = 442.5 := by
  sorry

end NUMINAMATH_CALUDE_debt_average_payment_l2033_203392


namespace NUMINAMATH_CALUDE_union_equality_implies_a_greater_than_one_l2033_203388

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 ≤ 1}
def B (a : ℝ) : Set ℝ := {x : ℝ | x < a}

-- State the theorem
theorem union_equality_implies_a_greater_than_one (a : ℝ) :
  A ∪ B a = B a → a > 1 := by
  sorry

end NUMINAMATH_CALUDE_union_equality_implies_a_greater_than_one_l2033_203388


namespace NUMINAMATH_CALUDE_specific_l_shape_perimeter_l2033_203352

/-- An L-shaped figure formed by squares -/
structure LShapedFigure where
  squareSideLength : ℕ
  baseSquares : ℕ
  stackedSquares : ℕ

/-- Calculate the perimeter of an L-shaped figure -/
def perimeter (figure : LShapedFigure) : ℕ :=
  2 * figure.squareSideLength * (figure.baseSquares + figure.stackedSquares + 1)

/-- Theorem: The perimeter of the specific L-shaped figure is 14 units -/
theorem specific_l_shape_perimeter :
  let figure : LShapedFigure := ⟨2, 3, 2⟩
  perimeter figure = 14 := by
  sorry

end NUMINAMATH_CALUDE_specific_l_shape_perimeter_l2033_203352


namespace NUMINAMATH_CALUDE_jacks_walking_speed_l2033_203311

/-- The problem of determining Jack's walking speed -/
theorem jacks_walking_speed
  (initial_distance : ℝ)
  (christina_speed : ℝ)
  (lindy_speed : ℝ)
  (lindy_distance : ℝ)
  (h1 : initial_distance = 270)
  (h2 : christina_speed = 5)
  (h3 : lindy_speed = 8)
  (h4 : lindy_distance = 240) :
  ∃ (jack_speed : ℝ),
    jack_speed = 4 ∧
    jack_speed * (lindy_distance / lindy_speed) +
    christina_speed * (lindy_distance / lindy_speed) =
    initial_distance :=
by sorry

end NUMINAMATH_CALUDE_jacks_walking_speed_l2033_203311


namespace NUMINAMATH_CALUDE_line_inclination_angle_l2033_203324

/-- The inclination angle of a line with point-slope form y - 2 = -√3(x - 1) is π/3 -/
theorem line_inclination_angle (x y : ℝ) :
  y - 2 = -Real.sqrt 3 * (x - 1) → ∃ α : ℝ, α = π / 3 ∧ Real.tan α = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_line_inclination_angle_l2033_203324


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l2033_203396

-- Problem 1
theorem problem_1 : 12 - (-18) - |(-7)| + 15 = 38 := by sorry

-- Problem 2
theorem problem_2 : -24 / (-3/2) + 6 * (-1/3) = 14 := by sorry

-- Problem 3
theorem problem_3 : (-7/9 + 5/6 - 1/4) * (-36) = 7 := by sorry

-- Problem 4
theorem problem_4 : -1^2 + 1/4 * (-2)^3 + (-3)^2 = 6 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l2033_203396


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2033_203365

theorem polynomial_simplification (x : ℝ) : 
  (3 * x^2 + 6 * x - 4) - (2 * x^2 + 3 * x - 15) = x^2 + 3 * x + 11 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2033_203365


namespace NUMINAMATH_CALUDE_library_books_existence_l2033_203355

theorem library_books_existence : ∃ (r P C B : ℕ), 
  r > 3000 ∧ 
  r = P + C + B ∧ 
  2 * P = 3 * C ∧ 
  3 * C = 4 * B :=
sorry

end NUMINAMATH_CALUDE_library_books_existence_l2033_203355


namespace NUMINAMATH_CALUDE_units_digit_of_sum_l2033_203372

-- Define a function to get the units digit of a number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Define the exponentiation operation for natural numbers
def pow (base : ℕ) (exp : ℕ) : ℕ := base ^ exp

-- Theorem statement
theorem units_digit_of_sum (a b c d : ℕ) :
  unitsDigit (pow a b + pow c d) = 9 :=
sorry

end NUMINAMATH_CALUDE_units_digit_of_sum_l2033_203372


namespace NUMINAMATH_CALUDE_lion_weight_is_41_3_l2033_203359

/-- The weight of a lion in kilograms -/
def lion_weight : ℝ := 41.3

/-- The weight of a tiger in kilograms -/
def tiger_weight : ℝ := lion_weight - 4.8

/-- The weight of a panda in kilograms -/
def panda_weight : ℝ := tiger_weight - 7.7

/-- Theorem stating that the weight of a lion is 41.3 kg given the conditions -/
theorem lion_weight_is_41_3 : 
  lion_weight = 41.3 ∧ 
  tiger_weight = lion_weight - 4.8 ∧
  panda_weight = tiger_weight - 7.7 ∧
  lion_weight + tiger_weight + panda_weight = 106.6 := by
  sorry

#check lion_weight_is_41_3

end NUMINAMATH_CALUDE_lion_weight_is_41_3_l2033_203359


namespace NUMINAMATH_CALUDE_geometric_series_calculation_l2033_203375

theorem geometric_series_calculation : 
  2016 * (1 / (1 + 1/2 + 1/4 + 1/8 + 1/16 + 1/32)) = 1024 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_calculation_l2033_203375


namespace NUMINAMATH_CALUDE_items_per_charge_l2033_203354

theorem items_per_charge (total_items : ℕ) (num_cards : ℕ) (h1 : total_items = 20) (h2 : num_cards = 4) :
  total_items / num_cards = 5 := by
sorry

end NUMINAMATH_CALUDE_items_per_charge_l2033_203354


namespace NUMINAMATH_CALUDE_horner_f_at_5_v2_eq_21_l2033_203336

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 2x^5 - 5x^4 - 4x^3 + 3x^2 - 6x + 7 -/
def f : List ℝ := [2, -5, -4, 3, -6, 7]

/-- Theorem: Horner's method for f(x) at x = 5 yields v_2 = 21 -/
theorem horner_f_at_5_v2_eq_21 :
  let v := horner f 5
  let v0 := 2
  let v1 := v0 * 5 - 5
  let v2 := v1 * 5 - 4
  v2 = 21 := by sorry

end NUMINAMATH_CALUDE_horner_f_at_5_v2_eq_21_l2033_203336


namespace NUMINAMATH_CALUDE_x_times_one_minus_f_equals_one_l2033_203332

/-- Given x = (3 + √8)^1001, n = ⌊x⌋, and f = x - n, prove that x(1 - f) = 1 -/
theorem x_times_one_minus_f_equals_one :
  let x : ℝ := (3 + Real.sqrt 8) ^ 1001
  let n : ℤ := ⌊x⌋
  let f : ℝ := x - n
  x * (1 - f) = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_times_one_minus_f_equals_one_l2033_203332


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l2033_203361

def M : Set ℝ := {x | x^2 - x = 0}
def N : Set ℝ := {y | y^2 + y = 0}

theorem union_of_M_and_N : M ∪ N = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l2033_203361


namespace NUMINAMATH_CALUDE_triangle_angle_C_l2033_203315

theorem triangle_angle_C (A B C : Real) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧  -- Sum of angles in a triangle
  3 * Real.sin A + 4 * Real.cos B = 6 ∧  -- Given condition
  4 * Real.sin B + 3 * Real.cos A = 1  -- Given condition
  → C = π / 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_C_l2033_203315


namespace NUMINAMATH_CALUDE_vote_count_theorem_l2033_203327

theorem vote_count_theorem (votes_A votes_B : ℕ) : 
  (votes_B = (20 * votes_A) / 21) →  -- B's votes are 20/21 of A's
  (votes_A > votes_B) →  -- A wins
  (votes_B + 4 > votes_A - 4) →  -- If B gains 4 votes, B would win
  (votes_A < 168) →  -- derived from the inequality in the solution
  (∀ (x : ℕ), x < votes_A → x ≠ votes_A ∨ (20 * x) / 21 ≠ votes_B) →  -- A's vote count is minimal
  ((votes_A = 147 ∧ votes_B = 140) ∨ (votes_A = 126 ∧ votes_B = 120)) :=
by
  sorry  -- Proof omitted

end NUMINAMATH_CALUDE_vote_count_theorem_l2033_203327


namespace NUMINAMATH_CALUDE_total_bricks_used_l2033_203398

/-- The number of brick walls -/
def total_walls : ℕ := 10

/-- The number of walls of the first type -/
def first_type_walls : ℕ := 5

/-- The number of walls of the second type -/
def second_type_walls : ℕ := 5

/-- The number of bricks in a single row for the first type of wall -/
def first_type_bricks_per_row : ℕ := 60

/-- The number of rows in the first type of wall -/
def first_type_rows : ℕ := 100

/-- The number of bricks in a single row for the second type of wall -/
def second_type_bricks_per_row : ℕ := 80

/-- The number of rows in the second type of wall -/
def second_type_rows : ℕ := 120

/-- Theorem: The total number of bricks used for all ten walls is 78000 -/
theorem total_bricks_used : 
  first_type_walls * first_type_bricks_per_row * first_type_rows +
  second_type_walls * second_type_bricks_per_row * second_type_rows = 78000 :=
by
  sorry

end NUMINAMATH_CALUDE_total_bricks_used_l2033_203398


namespace NUMINAMATH_CALUDE_inequality_condition_l2033_203340

theorem inequality_condition : 
  (∀ x : ℝ, -3 < x ∧ x < 0 → (x + 3) * (x - 2) < 0) ∧ 
  (∃ x : ℝ, (x + 3) * (x - 2) < 0 ∧ ¬(-3 < x ∧ x < 0)) :=
sorry

end NUMINAMATH_CALUDE_inequality_condition_l2033_203340


namespace NUMINAMATH_CALUDE_root_sum_squares_l2033_203339

/-- The polynomial p(x) = 4x^3 - 2x^2 - 15x + 9 -/
def p (x : ℝ) : ℝ := 4 * x^3 - 2 * x^2 - 15 * x + 9

/-- The polynomial q(x) = 12x^3 + 6x^2 - 7x + 1 -/
def q (x : ℝ) : ℝ := 12 * x^3 + 6 * x^2 - 7 * x + 1

/-- A is the largest root of p(x) -/
def A : ℝ := sorry

/-- B is the largest root of q(x) -/
def B : ℝ := sorry

/-- p(x) has exactly three distinct real roots -/
axiom p_has_three_roots : ∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
  (∀ (w : ℝ), p w = 0 ↔ w = x ∨ w = y ∨ w = z)

/-- q(x) has exactly three distinct real roots -/
axiom q_has_three_roots : ∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
  (∀ (w : ℝ), q w = 0 ↔ w = x ∨ w = y ∨ w = z)

/-- A is a root of p(x) -/
axiom A_is_root_of_p : p A = 0

/-- B is a root of q(x) -/
axiom B_is_root_of_q : q B = 0

/-- A is the largest root of p(x) -/
axiom A_is_largest_root_of_p : ∀ (x : ℝ), p x = 0 → x ≤ A

/-- B is the largest root of q(x) -/
axiom B_is_largest_root_of_q : ∀ (x : ℝ), q x = 0 → x ≤ B

theorem root_sum_squares : A^2 + 3 * B^2 = 4 := by sorry

end NUMINAMATH_CALUDE_root_sum_squares_l2033_203339


namespace NUMINAMATH_CALUDE_root_of_quadratic_l2033_203341

theorem root_of_quadratic (x v : ℝ) : 
  x = (-15 - Real.sqrt 409) / 12 →
  v = -23 / 3 →
  6 * x^2 + 15 * x + v = 0 := by sorry

end NUMINAMATH_CALUDE_root_of_quadratic_l2033_203341


namespace NUMINAMATH_CALUDE_multiply_power_equals_power_sum_problem_solution_l2033_203385

theorem multiply_power_equals_power_sum (a : ℕ) (m n : ℕ) : 
  a * (a^n) = a^(n + 1) := by sorry

theorem problem_solution : 
  3000 * (3000^3000) = 3000^3001 := by sorry

end NUMINAMATH_CALUDE_multiply_power_equals_power_sum_problem_solution_l2033_203385


namespace NUMINAMATH_CALUDE_number_count_proof_l2033_203320

theorem number_count_proof (total_avg : ℝ) (pair1_avg pair2_avg pair3_avg : ℝ) :
  total_avg = 3.95 →
  pair1_avg = 3.4 →
  pair2_avg = 3.85 →
  pair3_avg = 4.600000000000001 →
  (2 * pair1_avg + 2 * pair2_avg + 2 * pair3_avg) / total_avg = 6 := by
  sorry

#check number_count_proof

end NUMINAMATH_CALUDE_number_count_proof_l2033_203320


namespace NUMINAMATH_CALUDE_rectangle_circle_chord_length_l2033_203381

theorem rectangle_circle_chord_length :
  ∀ (rectangle : Set (ℝ × ℝ)) (circle : Set (ℝ × ℝ)) (P Q : ℝ × ℝ),
    -- Rectangle properties
    (∀ (x y : ℝ), (x, y) ∈ rectangle ↔ 0 ≤ x ∧ x ≤ 4 ∧ 0 ≤ y ∧ y ≤ 2) →
    -- Circle properties
    (∃ (cx cy : ℝ), ∀ (x y : ℝ), (x, y) ∈ circle ↔ (x - cx)^2 + (y - cy)^2 = 1) →
    -- Circle touches three sides of the rectangle
    (∃ (x : ℝ), (x, 0) ∈ circle ∧ 0 < x ∧ x < 4) →
    (∃ (y : ℝ), (0, y) ∈ circle ∧ 0 < y ∧ y < 2) →
    (∃ (x : ℝ), (x, 2) ∈ circle ∧ 0 < x ∧ x < 4) →
    -- P and Q are on the circle and the diagonal
    P ∈ circle → Q ∈ circle →
    (∃ (t : ℝ), 0 < t ∧ t < 1 ∧ P = (4*t, 2*t) ∧ Q = (4*(1-t), 2*(1-t))) →
    -- Conclusion: length of PQ is 4/√5
    Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 4 / Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_circle_chord_length_l2033_203381


namespace NUMINAMATH_CALUDE_anniversary_day_probability_probability_distribution_l2033_203346

def is_leap_year (year : ℕ) : Bool :=
  year % 4 = 0

def days_in_year (year : ℕ) : ℕ :=
  if is_leap_year year then 366 else 365

def days_between (start_year end_year : ℕ) : ℕ :=
  (List.range (end_year - start_year + 1)).foldl (λ acc y ↦ acc + days_in_year (start_year + y)) 0

theorem anniversary_day_probability (meeting_year : ℕ) 
  (h1 : meeting_year ≥ 1668 ∧ meeting_year ≤ 1671) :
  let total_days := days_between meeting_year (meeting_year + 11)
  let day_shift := total_days % 7
  (day_shift = 0 ∧ meeting_year ∈ [1668, 1670, 1671]) ∨
  (day_shift = 6 ∧ meeting_year = 1669) :=
sorry

theorem probability_distribution :
  let meeting_years := [1668, 1669, 1670, 1671]
  let friday_probability := (meeting_years.filter (λ y ↦ (days_between y (y + 11)) % 7 = 0)).length / meeting_years.length
  let thursday_probability := (meeting_years.filter (λ y ↦ (days_between y (y + 11)) % 7 = 6)).length / meeting_years.length
  friday_probability = 3/4 ∧ thursday_probability = 1/4 :=
sorry

end NUMINAMATH_CALUDE_anniversary_day_probability_probability_distribution_l2033_203346


namespace NUMINAMATH_CALUDE_solution_to_linear_equation_l2033_203397

theorem solution_to_linear_equation :
  ∃ (x y : ℝ), x + 2 * y = 6 ∧ x = 2 ∧ y = 2 := by
sorry

end NUMINAMATH_CALUDE_solution_to_linear_equation_l2033_203397


namespace NUMINAMATH_CALUDE_multiplication_equality_l2033_203337

-- Define the digits as natural numbers
def A : ℕ := 6
def B : ℕ := 7
def C : ℕ := 4
def D : ℕ := 2
def E : ℕ := 5
def F : ℕ := 9
def H : ℕ := 3
def J : ℕ := 8

-- Define the numbers ABCD and EF
def ABCD : ℕ := A * 1000 + B * 100 + C * 10 + D
def EF : ℕ := E * 10 + F

-- Define the result HFBBBJ
def HFBBBJ : ℕ := H * 100000 + F * 10000 + B * 1000 + B * 100 + B * 10 + J

-- State the theorem
theorem multiplication_equality :
  ABCD * EF = HFBBBJ :=
sorry

end NUMINAMATH_CALUDE_multiplication_equality_l2033_203337
