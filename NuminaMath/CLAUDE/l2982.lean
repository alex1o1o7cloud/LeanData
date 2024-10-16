import Mathlib

namespace NUMINAMATH_CALUDE_gcd_of_390_455_546_l2982_298252

theorem gcd_of_390_455_546 : Nat.gcd 390 (Nat.gcd 455 546) = 13 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_390_455_546_l2982_298252


namespace NUMINAMATH_CALUDE_jessica_jelly_bean_guess_l2982_298218

/-- Represents the number of jelly beans of each color in a bag -/
structure JellyBeanBag where
  red : ℕ
  black : ℕ
  green : ℕ
  purple : ℕ
  yellow : ℕ
  white : ℕ

/-- Calculates the number of bags needed to fill the fishbowl -/
def bagsNeeded (bag : JellyBeanBag) (guessRedWhite : ℕ) : ℕ :=
  guessRedWhite / (bag.red + bag.white)

theorem jessica_jelly_bean_guess 
  (bag : JellyBeanBag)
  (guessRedWhite : ℕ)
  (h1 : bag.red = 24)
  (h2 : bag.black = 13)
  (h3 : bag.green = 36)
  (h4 : bag.purple = 28)
  (h5 : bag.yellow = 32)
  (h6 : bag.white = 18)
  (h7 : guessRedWhite = 126) :
  bagsNeeded bag guessRedWhite = 3 := by
  sorry

end NUMINAMATH_CALUDE_jessica_jelly_bean_guess_l2982_298218


namespace NUMINAMATH_CALUDE_range_of_a_l2982_298240

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if two points are on opposite sides of a line given by 3x - 2y - a = 0 -/
def oppositeSides (p1 p2 : Point) (a : ℝ) : Prop :=
  (3 * p1.x - 2 * p1.y - a) * (3 * p2.x - 2 * p2.y - a) < 0

/-- The theorem stating the range of a given the conditions -/
theorem range_of_a (p1 p2 : Point) (h : oppositeSides p1 p2 a) :
  -8 < a ∧ a < 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2982_298240


namespace NUMINAMATH_CALUDE_power_of_five_division_l2982_298215

theorem power_of_five_division : (5 ^ 12) / (25 ^ 3) = 15625 := by sorry

end NUMINAMATH_CALUDE_power_of_five_division_l2982_298215


namespace NUMINAMATH_CALUDE_article_cost_calculation_l2982_298274

/-- Proves that if an article is sold for $25 with a 25% gain, it was bought for $20. -/
theorem article_cost_calculation (selling_price : ℝ) (gain_percent : ℝ) : 
  selling_price = 25 → gain_percent = 25 → 
  ∃ (cost_price : ℝ), cost_price = 20 ∧ selling_price = cost_price * (1 + gain_percent / 100) :=
by sorry

end NUMINAMATH_CALUDE_article_cost_calculation_l2982_298274


namespace NUMINAMATH_CALUDE_tan_seventeen_pi_fourths_l2982_298249

theorem tan_seventeen_pi_fourths : Real.tan (17 * Real.pi / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_seventeen_pi_fourths_l2982_298249


namespace NUMINAMATH_CALUDE_shopkeeper_profit_l2982_298283

/-- Proves that if a shopkeeper sells an article with a 4% discount and earns a 20% profit,
    then the profit percentage without discount would be 25%. -/
theorem shopkeeper_profit (cost_price : ℝ) (cost_price_pos : 0 < cost_price) :
  let discount_rate : ℝ := 0.04
  let profit_rate_with_discount : ℝ := 0.20
  let selling_price_with_discount : ℝ := cost_price * (1 + profit_rate_with_discount)
  let marked_price : ℝ := selling_price_with_discount / (1 - discount_rate)
  let profit_rate_without_discount : ℝ := (marked_price - cost_price) / cost_price
  profit_rate_without_discount = 0.25 := by
sorry

end NUMINAMATH_CALUDE_shopkeeper_profit_l2982_298283


namespace NUMINAMATH_CALUDE_complex_power_2019_l2982_298232

-- Define the imaginary unit i
variable (i : ℂ)

-- Define the property of i being the imaginary unit
axiom i_squared : i^2 = -1

-- State the theorem
theorem complex_power_2019 : (((1 + i) / (1 - i)) ^ 2019 : ℂ) = -i := by sorry

end NUMINAMATH_CALUDE_complex_power_2019_l2982_298232


namespace NUMINAMATH_CALUDE_initial_amount_of_A_l2982_298234

/-- Represents the money exchange problem with three people --/
structure MoneyExchange where
  a : ℕ  -- Initial amount of A
  b : ℕ  -- Initial amount of B
  c : ℕ  -- Initial amount of C

/-- Predicate that checks if the money exchange satisfies the problem conditions --/
def satisfies_conditions (m : MoneyExchange) : Prop :=
  -- After all exchanges, everyone has 16 dollars
  4 * (m.a - m.b - m.c) = 16 ∧
  6 * m.b - 2 * m.a - 2 * m.c = 16 ∧
  7 * m.c - m.a - m.b = 16

/-- Theorem stating that if the conditions are satisfied, A's initial amount was 29 --/
theorem initial_amount_of_A (m : MoneyExchange) :
  satisfies_conditions m → m.a = 29 := by
  sorry


end NUMINAMATH_CALUDE_initial_amount_of_A_l2982_298234


namespace NUMINAMATH_CALUDE_trig_identity_l2982_298273

theorem trig_identity (α : Real) (h : Real.tan α = 4) : 
  (1 + Real.cos (2 * α) + 8 * Real.sin α ^ 2) / Real.sin (2 * α) = 65 / 4 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2982_298273


namespace NUMINAMATH_CALUDE_total_notebooks_bought_l2982_298275

/-- Represents the number of notebooks in a large pack -/
def large_pack_size : ℕ := 7

/-- Represents the number of large packs Wilson bought -/
def large_packs_bought : ℕ := 7

/-- Theorem stating that the total number of notebooks Wilson bought is 49 -/
theorem total_notebooks_bought : large_pack_size * large_packs_bought = 49 := by
  sorry

end NUMINAMATH_CALUDE_total_notebooks_bought_l2982_298275


namespace NUMINAMATH_CALUDE_max_value_constraint_l2982_298267

theorem max_value_constraint (m : ℝ) : m > 1 →
  (∃ (x y : ℝ), y ≥ x ∧ y ≤ m * x ∧ x + y ≤ 1 ∧ x + m * y < 2) ↔ m < 1 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_constraint_l2982_298267


namespace NUMINAMATH_CALUDE_debt_payment_problem_l2982_298226

/-- Proves that the amount of each of the first 20 payments is $410 given the problem conditions. -/
theorem debt_payment_problem (total_payments : ℕ) (first_payments : ℕ) (payment_increase : ℕ) (average_payment : ℕ) :
  total_payments = 65 →
  first_payments = 20 →
  payment_increase = 65 →
  average_payment = 455 →
  ∃ (x : ℕ),
    x * first_payments + (x + payment_increase) * (total_payments - first_payments) = average_payment * total_payments ∧
    x = 410 :=
by sorry

end NUMINAMATH_CALUDE_debt_payment_problem_l2982_298226


namespace NUMINAMATH_CALUDE_zoo_animal_count_l2982_298290

theorem zoo_animal_count (initial_count : ℕ) (gorillas_sent : ℕ) (hippo_adopted : ℕ) 
  (rhinos_taken : ℕ) (final_count : ℕ) : 
  initial_count = 68 →
  gorillas_sent = 6 →
  hippo_adopted = 1 →
  rhinos_taken = 3 →
  final_count = 90 →
  ∃ (lion_cubs : ℕ), 
    final_count = initial_count - gorillas_sent + hippo_adopted + rhinos_taken + lion_cubs + 2 * lion_cubs ∧
    lion_cubs = 8 := by
  sorry

end NUMINAMATH_CALUDE_zoo_animal_count_l2982_298290


namespace NUMINAMATH_CALUDE_gcd_12345_23456_34567_l2982_298270

theorem gcd_12345_23456_34567 : Nat.gcd 12345 (Nat.gcd 23456 34567) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_12345_23456_34567_l2982_298270


namespace NUMINAMATH_CALUDE_smallest_s_for_triangle_l2982_298284

/-- A function that checks if three lengths can form a triangle -/
def can_form_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

/-- The theorem stating that 6 is the smallest whole number s such that 8, 13, and s can form a triangle -/
theorem smallest_s_for_triangle : 
  ∀ s : ℕ, s ≥ 6 ↔ can_form_triangle 8 13 s ∧ 
  (∀ t : ℕ, t < s → ¬can_form_triangle 8 13 t) :=
by sorry

#check smallest_s_for_triangle

end NUMINAMATH_CALUDE_smallest_s_for_triangle_l2982_298284


namespace NUMINAMATH_CALUDE_calculator_key_presses_l2982_298229

def f (x : ℕ) : ℕ := x^2 - 3

def iterate_f (n : ℕ) (x : ℕ) : ℕ :=
  match n with
  | 0 => x
  | n+1 => f (iterate_f n x)

theorem calculator_key_presses :
  iterate_f 2 4 ≤ 2000 ∧ iterate_f 3 4 > 2000 := by
  sorry

end NUMINAMATH_CALUDE_calculator_key_presses_l2982_298229


namespace NUMINAMATH_CALUDE_eight_bead_necklace_arrangements_l2982_298282

/-- The number of distinct arrangements of beads on a necklace with specific properties. -/
def necklaceArrangements (n : ℕ) : ℕ :=
  Nat.factorial n / 2

/-- Theorem stating that the number of distinct arrangements of 8 beads
    on a necklace with a fixed pendant and reflectional symmetry is 8! / 2. -/
theorem eight_bead_necklace_arrangements :
  necklaceArrangements 8 = 20160 := by
  sorry

#eval necklaceArrangements 8

end NUMINAMATH_CALUDE_eight_bead_necklace_arrangements_l2982_298282


namespace NUMINAMATH_CALUDE_like_terms_imply_value_l2982_298295

theorem like_terms_imply_value (m n : ℤ) : 
  (m + 2 = 6 ∧ n + 1 = 3) → (-m)^3 + n^2 = -60 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_imply_value_l2982_298295


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2982_298214

/-- Sum of the first n terms of a geometric sequence -/
def geometricSum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The problem statement -/
theorem geometric_sequence_sum :
  let a : ℚ := 1/6
  let r : ℚ := 1/2
  let n : ℕ := 6
  geometricSum a r n = 21/64 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2982_298214


namespace NUMINAMATH_CALUDE_intersection_equals_interval_l2982_298298

-- Define sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 2*x - 8 < 0}
def B : Set ℝ := {x : ℝ | x ≥ 0}

-- Define the interval [0, 4)
def interval : Set ℝ := {x : ℝ | 0 ≤ x ∧ x < 4}

-- Theorem statement
theorem intersection_equals_interval : A ∩ B = interval := by sorry

end NUMINAMATH_CALUDE_intersection_equals_interval_l2982_298298


namespace NUMINAMATH_CALUDE_certain_event_l2982_298280

/-- Represents the color of a ball -/
inductive Color
| Red
| White

/-- Represents a bag of balls -/
def Bag := List Color

/-- Represents the result of drawing two balls -/
def Draw := (Color × Color)

/-- The bag containing 2 red balls and 1 white ball -/
def initialBag : Bag := [Color.Red, Color.Red, Color.White]

/-- Function to check if a draw contains at least one red ball -/
def hasRed (draw : Draw) : Prop :=
  draw.1 = Color.Red ∨ draw.2 = Color.Red

/-- All possible draws from the bag -/
def allDraws : List Draw := [
  (Color.Red, Color.Red),
  (Color.Red, Color.White),
  (Color.White, Color.Red)
]

/-- Theorem stating that any draw from the bag must contain at least one red ball -/
theorem certain_event : ∀ (draw : Draw), draw ∈ allDraws → hasRed draw := by sorry

end NUMINAMATH_CALUDE_certain_event_l2982_298280


namespace NUMINAMATH_CALUDE_simplify_sqrt_m3n2_l2982_298210

theorem simplify_sqrt_m3n2 (m n : ℝ) (hm : m > 0) (hn : n < 0) :
  Real.sqrt (m^3 * n^2) = -m * n * Real.sqrt m := by sorry

end NUMINAMATH_CALUDE_simplify_sqrt_m3n2_l2982_298210


namespace NUMINAMATH_CALUDE_third_grade_agreement_l2982_298269

theorem third_grade_agreement (total_agreed : ℕ) (fourth_grade_agreed : ℕ) 
  (h1 : total_agreed = 391) (h2 : fourth_grade_agreed = 237) :
  total_agreed - fourth_grade_agreed = 154 := by
  sorry

end NUMINAMATH_CALUDE_third_grade_agreement_l2982_298269


namespace NUMINAMATH_CALUDE_combined_bus_capacity_l2982_298296

/-- The capacity of the train -/
def train_capacity : ℕ := 120

/-- The number of buses -/
def num_buses : ℕ := 2

/-- The capacity of one bus as a fraction of the train's capacity -/
def bus_capacity_fraction : ℚ := 1 / 6

/-- Theorem: The combined capacity of the two buses is 40 people -/
theorem combined_bus_capacity :
  (num_buses : ℚ) * (bus_capacity_fraction * train_capacity) = 40 := by
  sorry

end NUMINAMATH_CALUDE_combined_bus_capacity_l2982_298296


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2982_298200

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_of_M_and_N : M ∩ N = {2, 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2982_298200


namespace NUMINAMATH_CALUDE_transportation_charges_proof_l2982_298299

def transportation_charges (purchase_price repair_cost profit_percentage actual_selling_price : ℕ) : ℕ :=
  let total_cost_before_transport := purchase_price + repair_cost
  let profit := (total_cost_before_transport * profit_percentage) / 100
  let calculated_selling_price := total_cost_before_transport + profit
  actual_selling_price - calculated_selling_price

theorem transportation_charges_proof :
  transportation_charges 9000 5000 50 22500 = 1500 := by
  sorry

end NUMINAMATH_CALUDE_transportation_charges_proof_l2982_298299


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2982_298243

/-- The imaginary part of 1 / (1 + i) is -1/2 -/
theorem imaginary_part_of_z (z : ℂ) : z = 1 / (1 + Complex.I) → z.im = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2982_298243


namespace NUMINAMATH_CALUDE_absolute_difference_of_U_coordinates_l2982_298276

/-- Triangle PQR with vertices P(0,10), Q(5,0), and R(10,0) -/
def P : ℝ × ℝ := (0, 10)
def Q : ℝ × ℝ := (5, 0)
def R : ℝ × ℝ := (10, 0)

/-- V is on QR and 3 units away from Q -/
def V : ℝ × ℝ := (2, 0)

/-- U is on PR and has the same x-coordinate as V -/
def U : ℝ × ℝ := (2, 8)

/-- The theorem to be proved -/
theorem absolute_difference_of_U_coordinates : 
  |U.2 - U.1| = 6 := by sorry

end NUMINAMATH_CALUDE_absolute_difference_of_U_coordinates_l2982_298276


namespace NUMINAMATH_CALUDE_sequence_perfect_squares_l2982_298242

theorem sequence_perfect_squares (a b : ℕ → ℤ) :
  (∀ n : ℕ, a (n + 1) = 7 * a n + 6 * b n - 3) →
  (∀ n : ℕ, b (n + 1) = 8 * a n + 7 * b n - 4) →
  ∃ A : ℕ → ℤ, ∀ n : ℕ, a n = (A n)^2 := by
sorry

end NUMINAMATH_CALUDE_sequence_perfect_squares_l2982_298242


namespace NUMINAMATH_CALUDE_crafts_club_members_crafts_club_members_proof_l2982_298258

theorem crafts_club_members : ℕ → Prop :=
  fun n =>
    let necklaces_per_member : ℕ := 2
    let beads_per_necklace : ℕ := 50
    let total_beads : ℕ := 900
    n * (necklaces_per_member * beads_per_necklace) = total_beads →
    n = 9

-- Proof
theorem crafts_club_members_proof : crafts_club_members 9 := by
  sorry

end NUMINAMATH_CALUDE_crafts_club_members_crafts_club_members_proof_l2982_298258


namespace NUMINAMATH_CALUDE_condition_t_necessary_not_sufficient_l2982_298216

theorem condition_t_necessary_not_sufficient (x y : ℝ) :
  (∀ x y, (x + y ≤ 28 ∨ x * y ≤ 192) → (x ≤ 12 ∨ y ≤ 16)) ∧
  (∃ x y, (x ≤ 12 ∨ y ≤ 16) ∧ ¬(x + y ≤ 28 ∨ x * y ≤ 192)) :=
by sorry

end NUMINAMATH_CALUDE_condition_t_necessary_not_sufficient_l2982_298216


namespace NUMINAMATH_CALUDE_alcohol_solution_proof_l2982_298223

/-- Proves that adding 2.4 liters of pure alcohol to a 6-liter solution that is 30% alcohol
    will result in a solution that is 50% alcohol. -/
theorem alcohol_solution_proof (initial_volume : ℝ) (initial_concentration : ℝ)
  (added_alcohol : ℝ) (target_concentration : ℝ)
  (h1 : initial_volume = 6)
  (h2 : initial_concentration = 0.3)
  (h3 : added_alcohol = 2.4)
  (h4 : target_concentration = 0.5) :
  (initial_volume * initial_concentration + added_alcohol) / (initial_volume + added_alcohol) = target_concentration :=
by sorry

end NUMINAMATH_CALUDE_alcohol_solution_proof_l2982_298223


namespace NUMINAMATH_CALUDE_physics_marks_proof_l2982_298205

def english_marks : ℕ := 96
def math_marks : ℕ := 95
def chemistry_marks : ℕ := 87
def biology_marks : ℕ := 92
def average_marks : ℚ := 90.4
def total_subjects : ℕ := 5

theorem physics_marks_proof :
  ∃ (physics_marks : ℕ),
    (english_marks + math_marks + chemistry_marks + biology_marks + physics_marks : ℚ) / total_subjects = average_marks ∧
    physics_marks = 82 := by
  sorry

end NUMINAMATH_CALUDE_physics_marks_proof_l2982_298205


namespace NUMINAMATH_CALUDE_product_equality_l2982_298237

theorem product_equality : 2.05 * 4.1 = 20.5 * 0.41 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l2982_298237


namespace NUMINAMATH_CALUDE_nancys_weight_calculation_l2982_298292

/-- Nancy's weight in pounds -/
def nancys_weight : ℝ := 90

/-- Nancy's daily water intake as a percentage of her body weight -/
def water_intake_percentage : ℝ := 60

/-- Nancy's daily water intake in pounds -/
def daily_water_intake : ℝ := 54

theorem nancys_weight_calculation :
  nancys_weight * (water_intake_percentage / 100) = daily_water_intake :=
by sorry

end NUMINAMATH_CALUDE_nancys_weight_calculation_l2982_298292


namespace NUMINAMATH_CALUDE_smallest_difference_in_triangle_l2982_298260

theorem smallest_difference_in_triangle (a b c : ℕ) : 
  a + b + c = 2023 →
  a < b →
  b ≤ c →
  (∀ x y z : ℕ, x + y + z = 2023 → x < y → y ≤ z → b - a ≤ y - x) →
  b - a = 1 := by
sorry

end NUMINAMATH_CALUDE_smallest_difference_in_triangle_l2982_298260


namespace NUMINAMATH_CALUDE_vector_inequality_l2982_298285

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define vectors a, b, c, d in the vector space V
variable (a b c d : V)

-- Define the condition that the sum of vectors is zero
def sum_is_zero : Prop := a + b + c + d = 0

-- State the theorem
theorem vector_inequality (h : sum_is_zero a b c d) : 
  ‖a‖ + ‖b‖ + ‖c‖ + ‖d‖ ≥ ‖a + d‖ + ‖b + d‖ + ‖c + d‖ := by
  sorry

end NUMINAMATH_CALUDE_vector_inequality_l2982_298285


namespace NUMINAMATH_CALUDE_flash_catches_ace_l2982_298221

/-- The distance Flash must run to catch Ace -/
def flashDistance (x v c y : ℝ) : ℝ := 2 * y

theorem flash_catches_ace (x v c y : ℝ) 
  (hx : x > 1) 
  (hc : c > 0) : 
  flashDistance x v c y = 2 * y := by
  sorry

#check flash_catches_ace

end NUMINAMATH_CALUDE_flash_catches_ace_l2982_298221


namespace NUMINAMATH_CALUDE_z_equation_l2982_298203

theorem z_equation (x y z : ℝ) (h : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ x + y - 2*x*y ≠ 0) :
  (1/x + 1/y = 2 + 1/z) → z = (x*y)/(x + y - 2*x*y) := by
  sorry

end NUMINAMATH_CALUDE_z_equation_l2982_298203


namespace NUMINAMATH_CALUDE_cube_sum_divisibility_l2982_298236

theorem cube_sum_divisibility (a₁ a₂ a₃ a₄ a₅ : ℤ) :
  (9 ∣ (a₁^3 + a₂^3 + a₃^3 + a₄^3 + a₅^3)) → (3 ∣ (a₁ * a₂ * a₃ * a₄ * a₅)) := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_divisibility_l2982_298236


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l2982_298254

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_formula 
  (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a)
  (h_sum1 : a 1 + a 2 + a 3 = 0)
  (h_sum2 : a 4 + a 5 + a 6 = 18) :
  ∀ n : ℕ, a n = 2 * n - 4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l2982_298254


namespace NUMINAMATH_CALUDE_platform_length_l2982_298201

/-- The length of a platform given a train's speed and passing times -/
theorem platform_length (train_speed : ℝ) (platform_time : ℝ) (man_time : ℝ) :
  train_speed = 54 →
  platform_time = 30 →
  man_time = 20 →
  (train_speed * 1000 / 3600) * platform_time - (train_speed * 1000 / 3600) * man_time = 150 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_l2982_298201


namespace NUMINAMATH_CALUDE_undeveloped_land_area_l2982_298217

theorem undeveloped_land_area (total_area : ℝ) (num_sections : ℕ) 
  (h1 : total_area = 7305)
  (h2 : num_sections = 3) :
  total_area / num_sections = 2435 := by
  sorry

end NUMINAMATH_CALUDE_undeveloped_land_area_l2982_298217


namespace NUMINAMATH_CALUDE_line_parameterization_l2982_298209

-- Define the line equation
def line_equation (x y : ℝ) : Prop := y = 4 * x - 9

-- Define the parameterization
def parameterization (x y s p t : ℝ) : Prop :=
  x = s + 5 * t ∧ y = 3 + p * t

-- Theorem statement
theorem line_parameterization (s p : ℝ) :
  (∀ x y t : ℝ, line_equation x y ∧ parameterization x y s p t) →
  s = 3 ∧ p = 20 := by sorry

end NUMINAMATH_CALUDE_line_parameterization_l2982_298209


namespace NUMINAMATH_CALUDE_four_roots_iff_t_in_range_l2982_298253

-- Define the function f(x) = |xe^x|
noncomputable def f (x : ℝ) : ℝ := |x * Real.exp x|

-- Define the equation f^2(x) + tf(x) + 2 = 0
def has_four_distinct_roots (t : ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ x₄ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    (f x₁)^2 + t * f x₁ + 2 = 0 ∧
    (f x₂)^2 + t * f x₂ + 2 = 0 ∧
    (f x₃)^2 + t * f x₃ + 2 = 0 ∧
    (f x₄)^2 + t * f x₄ + 2 = 0

-- The theorem to be proved
theorem four_roots_iff_t_in_range :
  ∀ t : ℝ, has_four_distinct_roots t ↔ t < -(2 * Real.exp 2 + 1) / Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_four_roots_iff_t_in_range_l2982_298253


namespace NUMINAMATH_CALUDE_mushroom_collection_l2982_298233

theorem mushroom_collection 
  (N I A V : ℝ) 
  (h_non_negative : 0 ≤ N ∧ 0 ≤ I ∧ 0 ≤ A ∧ 0 ≤ V)
  (h_natasha_most : N > I ∧ N > A ∧ N > V)
  (h_ira_least : I ≤ N ∧ I ≤ A ∧ I ≤ V)
  (h_alyosha_more : A > V) : 
  N + I > A + V := by
  sorry

end NUMINAMATH_CALUDE_mushroom_collection_l2982_298233


namespace NUMINAMATH_CALUDE_congruence_problem_l2982_298248

theorem congruence_problem (n : ℤ) : 
  0 ≤ n ∧ n < 103 ∧ (100 * n) % 103 = 85 % 103 → n % 103 = 6 % 103 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l2982_298248


namespace NUMINAMATH_CALUDE_correct_position_probability_l2982_298250

/-- The number of books -/
def n : ℕ := 9

/-- The number of books to be in the correct position -/
def k : ℕ := 6

/-- The probability of exactly k books being in their correct position when n books are randomly rearranged -/
def probability (n k : ℕ) : ℚ := sorry

theorem correct_position_probability : probability n k = 1 / 2160 := by sorry

end NUMINAMATH_CALUDE_correct_position_probability_l2982_298250


namespace NUMINAMATH_CALUDE_inequality_theorem_l2982_298244

theorem inequality_theorem (a b c : ℝ) : a > -b → c - a < c + b := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l2982_298244


namespace NUMINAMATH_CALUDE_two_points_determine_line_l2982_298266

-- Define a Point type
structure Point where
  x : ℝ
  y : ℝ

-- Define a Line type
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Function to check if a point lies on a line
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Theorem statement
theorem two_points_determine_line (p1 p2 : Point) (h : p1 ≠ p2) :
  ∃! l : Line, pointOnLine p1 l ∧ pointOnLine p2 l :=
sorry

end NUMINAMATH_CALUDE_two_points_determine_line_l2982_298266


namespace NUMINAMATH_CALUDE_no_real_roots_quadratic_l2982_298278

theorem no_real_roots_quadratic : ∀ x : ℝ, x^2 + 2*x + 2 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_quadratic_l2982_298278


namespace NUMINAMATH_CALUDE_complex_multiplication_l2982_298255

theorem complex_multiplication (i : ℂ) : i^2 = -1 → (2 - i) * (-2 + i) = -3 + 4*i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l2982_298255


namespace NUMINAMATH_CALUDE_exists_valid_arrangement_l2982_298265

/-- Represents a position on a regular 25-gon -/
inductive Position
| Vertex : Fin 25 → Position
| Midpoint : Fin 25 → Position

/-- Represents an arrangement of numbers on a regular 25-gon -/
def Arrangement := Position → Fin 50

/-- Checks if the sum of numbers at the ends and midpoint of a side is constant -/
def isConstantSum (arr : Arrangement) : Prop :=
  ∃ s : ℕ, ∀ i : Fin 25,
    (arr (Position.Vertex i)).val + 
    (arr (Position.Midpoint i)).val + 
    (arr (Position.Vertex ((i.val + 1) % 25 : Fin 25))).val = s

/-- Theorem stating the existence of a valid arrangement -/
theorem exists_valid_arrangement : ∃ arr : Arrangement, isConstantSum arr ∧ 
  (∀ p : Position, (arr p).val ≥ 1 ∧ (arr p).val ≤ 50) ∧
  (∀ p q : Position, p ≠ q → arr p ≠ arr q) :=
sorry

end NUMINAMATH_CALUDE_exists_valid_arrangement_l2982_298265


namespace NUMINAMATH_CALUDE_jessica_bank_balance_l2982_298289

theorem jessica_bank_balance (B : ℝ) : 
  B - 400 = (3/5) * B → 
  B - 400 + (1/4) * (B - 400) = 750 := by
sorry

end NUMINAMATH_CALUDE_jessica_bank_balance_l2982_298289


namespace NUMINAMATH_CALUDE_quadratic_rational_solutions_l2982_298287

/-- A function that checks if a number is a perfect square -/
def is_perfect_square (n : ℤ) : Prop := ∃ m : ℤ, n = m * m

/-- The discriminant of the quadratic equation kx^2 - 24x + 4k = 0 -/
def discriminant (k : ℤ) : ℤ := 576 - 16 * k * k

/-- The property that k is a valid solution -/
def is_valid_k (k : ℤ) : Prop :=
  k > 0 ∧ is_perfect_square (discriminant k)

theorem quadratic_rational_solutions :
  ∀ k : ℤ, is_valid_k k ↔ k = 3 ∨ k = 6 := by sorry

end NUMINAMATH_CALUDE_quadratic_rational_solutions_l2982_298287


namespace NUMINAMATH_CALUDE_average_percentages_correct_l2982_298251

-- Define the subjects
inductive Subject
  | English
  | Mathematics
  | Physics
  | Chemistry
  | Biology
  | History
  | Geography

-- Define the marks and total marks for each subject
def marks (s : Subject) : ℕ :=
  match s with
  | Subject.English => 76
  | Subject.Mathematics => 65
  | Subject.Physics => 82
  | Subject.Chemistry => 67
  | Subject.Biology => 85
  | Subject.History => 92
  | Subject.Geography => 58

def totalMarks (s : Subject) : ℕ :=
  match s with
  | Subject.English => 120
  | Subject.Mathematics => 150
  | Subject.Physics => 100
  | Subject.Chemistry => 80
  | Subject.Biology => 100
  | Subject.History => 150
  | Subject.Geography => 75

-- Define the average percentage calculation
def averagePercentage (s : Subject) : ℚ :=
  (marks s : ℚ) / (totalMarks s : ℚ) * 100

-- Theorem to prove the correctness of average percentages
theorem average_percentages_correct :
  averagePercentage Subject.English = 63.33 ∧
  averagePercentage Subject.Mathematics = 43.33 ∧
  averagePercentage Subject.Physics = 82 ∧
  averagePercentage Subject.Chemistry = 83.75 ∧
  averagePercentage Subject.Biology = 85 ∧
  averagePercentage Subject.History = 61.33 ∧
  averagePercentage Subject.Geography = 77.33 := by
  sorry


end NUMINAMATH_CALUDE_average_percentages_correct_l2982_298251


namespace NUMINAMATH_CALUDE_marie_stamps_l2982_298219

theorem marie_stamps (notebooks : ℕ) (stamps_per_notebook : ℕ) (binders : ℕ) (stamps_per_binder : ℕ) (stamps_given_away : ℕ) :
  notebooks = 4 →
  stamps_per_notebook = 20 →
  binders = 2 →
  stamps_per_binder = 50 →
  stamps_given_away = 135 →
  (notebooks * stamps_per_notebook + binders * stamps_per_binder - stamps_given_away : ℚ) / (notebooks * stamps_per_notebook + binders * stamps_per_binder) = 1 / 4 :=
by sorry

end NUMINAMATH_CALUDE_marie_stamps_l2982_298219


namespace NUMINAMATH_CALUDE_max_slope_no_lattice_points_l2982_298213

-- Define a lattice point
def is_lattice_point (x y : ℤ) : Prop := True

-- Define the line equation
def on_line (m : ℚ) (x y : ℤ) : Prop := y = m * x + 2

-- Define the condition for no lattice points
def no_lattice_points (m : ℚ) : Prop :=
  ∀ x y : ℤ, 0 < x ∧ x ≤ 100 → is_lattice_point x y → ¬(on_line m x y)

-- State the theorem
theorem max_slope_no_lattice_points :
  (∀ m : ℚ, 1/2 < m → m < 50/99 → no_lattice_points m) ∧
  ¬(∀ m : ℚ, 1/2 < m → m < 50/99 + ε → no_lattice_points m) :=
sorry

end NUMINAMATH_CALUDE_max_slope_no_lattice_points_l2982_298213


namespace NUMINAMATH_CALUDE_inequalities_truth_l2982_298288

theorem inequalities_truth (a b c d : ℝ) : 
  (a^2 + b^2 + c^2 ≥ a*b + b*c + a*c) ∧ 
  (a*(1 - a) ≤ (1/4 : ℝ)) ∧ 
  ((a^2 + b^2)*(c^2 + d^2) ≥ (a*c + b*d)^2) ∧
  ¬(∀ (a b : ℝ), a/b + b/a ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_truth_l2982_298288


namespace NUMINAMATH_CALUDE_clothes_cost_calculation_l2982_298235

def savings_june : ℕ := 21
def savings_july : ℕ := 46
def savings_august : ℕ := 45
def school_supplies_cost : ℕ := 12
def remaining_balance : ℕ := 46

def total_savings : ℕ := savings_june + savings_july + savings_august

def clothes_cost : ℕ := total_savings - school_supplies_cost - remaining_balance

theorem clothes_cost_calculation :
  clothes_cost = 54 :=
by sorry

end NUMINAMATH_CALUDE_clothes_cost_calculation_l2982_298235


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_area_and_perimeter_l2982_298222

-- Define an isosceles right triangle with rational hypotenuse
structure IsoscelesRightTriangle where
  hypotenuse : ℚ
  hypotenuse_positive : hypotenuse > 0

-- Define the area of the triangle
def area (t : IsoscelesRightTriangle) : ℚ :=
  t.hypotenuse ^ 2 / 4

-- Define the perimeter of the triangle
noncomputable def perimeter (t : IsoscelesRightTriangle) : ℝ :=
  t.hypotenuse * (2 + Real.sqrt 2)

-- Theorem statement
theorem isosceles_right_triangle_area_and_perimeter (t : IsoscelesRightTriangle) :
  (∃ q : ℚ, area t = q) ∧ (∀ q : ℚ, perimeter t ≠ q) := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_area_and_perimeter_l2982_298222


namespace NUMINAMATH_CALUDE_cos_sum_of_complex_exponentials_l2982_298257

theorem cos_sum_of_complex_exponentials (γ δ : ℝ) :
  Complex.exp (γ * I) = 4/5 + 3/5 * I →
  Complex.exp (δ * I) = -5/13 - 12/13 * I →
  Real.cos (γ + δ) = 16/65 := by
  sorry

end NUMINAMATH_CALUDE_cos_sum_of_complex_exponentials_l2982_298257


namespace NUMINAMATH_CALUDE_intersection_of_sets_l2982_298202

theorem intersection_of_sets (P Q : Set ℝ) : 
  (P = {y : ℝ | ∃ x : ℝ, y = x + 1}) → 
  (Q = {y : ℝ | ∃ x : ℝ, y = 1 - x}) → 
  P ∩ Q = Set.univ := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l2982_298202


namespace NUMINAMATH_CALUDE_expression_equals_six_l2982_298247

theorem expression_equals_six : 
  Real.sqrt 16 - 2 * Real.tan (45 * π / 180) + |(-3)| + (π - 2022) ^ (0 : ℕ) = 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_six_l2982_298247


namespace NUMINAMATH_CALUDE_two_students_adjacent_probability_l2982_298297

theorem two_students_adjacent_probability (n : ℕ) (h : n = 10) :
  (2 * Nat.factorial (n - 1)) / Nat.factorial n = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_two_students_adjacent_probability_l2982_298297


namespace NUMINAMATH_CALUDE_no_two_digit_prime_sum_9_div_3_l2982_298262

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

theorem no_two_digit_prime_sum_9_div_3 :
  ¬ ∃ (n : ℕ), is_two_digit n ∧ Nat.Prime n ∧ sum_of_digits n = 9 ∧ n % 3 = 0 :=
sorry

end NUMINAMATH_CALUDE_no_two_digit_prime_sum_9_div_3_l2982_298262


namespace NUMINAMATH_CALUDE_consecutive_odd_integers_problem_l2982_298279

theorem consecutive_odd_integers_problem (x : ℤ) (k : ℕ) : 
  (x % 2 = 1) →  -- x is odd
  ((x + 2) % 2 = 1) →  -- x + 2 is odd
  ((x + 4) % 2 = 1) →  -- x + 4 is odd
  (x + (x + 4) = k * (x + 2) - 131) →  -- sum of 1st and 3rd is 131 less than k times 2nd
  (x + (x + 2) + (x + 4) = 133) →  -- sum of all three is 133
  (k = 2) := by sorry

end NUMINAMATH_CALUDE_consecutive_odd_integers_problem_l2982_298279


namespace NUMINAMATH_CALUDE_positive_real_inequality_l2982_298256

theorem positive_real_inequality (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : x + y + z = 1/x + 1/y + 1/z) :
  x + y + z ≥ Real.sqrt ((x*y + 1)/2) + Real.sqrt ((y*z + 1)/2) + Real.sqrt ((z*x + 1)/2) :=
by sorry

end NUMINAMATH_CALUDE_positive_real_inequality_l2982_298256


namespace NUMINAMATH_CALUDE_operation_on_original_number_l2982_298277

theorem operation_on_original_number : ∃ (f : ℝ → ℝ), 
  (3 * (f 4 + 9) = 51) ∧ (f 4 = 2 * 4) := by
  sorry

end NUMINAMATH_CALUDE_operation_on_original_number_l2982_298277


namespace NUMINAMATH_CALUDE_tangent_parallel_to_x_axis_l2982_298228

/-- The function f(x) defined in the problem -/
def f (t : ℝ) (x : ℝ) : ℝ := x^3 + (2*t - 1)*x + 3

/-- The derivative of f(x) with respect to x -/
def f' (t : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*t - 1

/-- Theorem stating that t = -1 given the conditions -/
theorem tangent_parallel_to_x_axis (t : ℝ) : 
  (f' t (-1) = 0) → t = -1 := by
  sorry

#check tangent_parallel_to_x_axis

end NUMINAMATH_CALUDE_tangent_parallel_to_x_axis_l2982_298228


namespace NUMINAMATH_CALUDE_three_digit_square_sum_numbers_l2982_298263

def is_valid_number (n : ℕ) : Prop :=
  ∃ (a b c : ℕ),
    a ≠ 0 ∧
    a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9 ∧
    n = 100 * a + 10 * b + c ∧
    n = 11 * (a^2 + b^2 + c^2)

theorem three_digit_square_sum_numbers :
  {n : ℕ | is_valid_number n} = {550, 803} :=
by sorry

end NUMINAMATH_CALUDE_three_digit_square_sum_numbers_l2982_298263


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l2982_298227

open Set

def U : Set Nat := {0, 1, 2, 3, 4}
def M : Set Nat := {0, 1, 2}
def N : Set Nat := {2, 3}

theorem complement_intersection_theorem :
  (U \ M) ∩ N = {3} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l2982_298227


namespace NUMINAMATH_CALUDE_complex_difference_magnitude_l2982_298204

theorem complex_difference_magnitude (z₁ z₂ : ℂ) 
  (h1 : Complex.abs z₁ = 2)
  (h2 : Complex.abs z₂ = 2)
  (h3 : z₁ + z₂ = 1 + Complex.I * Real.sqrt 3) :
  Complex.abs (z₁ - z₂) = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_difference_magnitude_l2982_298204


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2982_298206

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x : ℝ, (3*x - 5)^7 = a₀ + a₁*(x-1) + a₂*(x-1)^2 + a₃*(x-1)^3 + a₄*(x-1)^4 + a₅*(x-1)^5 + a₆*(x-1)^6 + a₇*(x-1)^7) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 129 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2982_298206


namespace NUMINAMATH_CALUDE_card_sum_theorem_l2982_298286

theorem card_sum_theorem (a b c d e f g h : ℕ) :
  (a + b) * (c + d) * (e + f) * (g + h) = 330 →
  a + b + c + d + e + f + g + h = 21 := by
sorry

end NUMINAMATH_CALUDE_card_sum_theorem_l2982_298286


namespace NUMINAMATH_CALUDE_milk_bottles_count_l2982_298231

theorem milk_bottles_count (bread : ℕ) (total : ℕ) (h1 : bread = 37) (h2 : total = 52) :
  total - bread = 15 := by
  sorry

end NUMINAMATH_CALUDE_milk_bottles_count_l2982_298231


namespace NUMINAMATH_CALUDE_major_premise_for_increasing_cubic_l2982_298272

-- Define the function y = x³
def f (x : ℝ) : ℝ := x^3

-- Define what it means for a function to be increasing
def IsIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂

-- State the theorem
theorem major_premise_for_increasing_cubic :
  (∀ g : ℝ → ℝ, IsIncreasing g ↔ (∀ x₁ x₂ : ℝ, x₁ < x₂ → g x₁ < g x₂)) →
  IsIncreasing f :=
by sorry

end NUMINAMATH_CALUDE_major_premise_for_increasing_cubic_l2982_298272


namespace NUMINAMATH_CALUDE_trig_identity_l2982_298294

theorem trig_identity (α : Real) (h : 3 * Real.sin α + Real.cos α = 0) :
  1 / (Real.cos α ^ 2 + 2 * Real.sin α * Real.cos α) = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2982_298294


namespace NUMINAMATH_CALUDE_color_one_third_square_l2982_298245

theorem color_one_third_square (n : ℕ) (k : ℕ) : n = 18 → k = 6 → Nat.choose n k = 18564 := by
  sorry

end NUMINAMATH_CALUDE_color_one_third_square_l2982_298245


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2982_298271

theorem arithmetic_sequence_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) (aₙ : ℕ) :
  a₁ = 1 →
  d = 2 →
  n > 0 →
  aₙ = 21 →
  aₙ = a₁ + (n - 1) * d →
  (n * (a₁ + aₙ)) / 2 = 121 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2982_298271


namespace NUMINAMATH_CALUDE_price_after_discounts_l2982_298239

/-- The original price of an article before discounts -/
def original_price : ℝ := 50

/-- The first discount rate -/
def discount1 : ℝ := 0.1

/-- The second discount rate -/
def discount2 : ℝ := 0.2

/-- The final sale price after discounts -/
def final_price : ℝ := 36

theorem price_after_discounts :
  original_price * (1 - discount1) * (1 - discount2) = final_price := by
  sorry

end NUMINAMATH_CALUDE_price_after_discounts_l2982_298239


namespace NUMINAMATH_CALUDE_correct_calculation_l2982_298230

theorem correct_calculation (x : ℤ) (h : x + 65 = 125) : x + 95 = 155 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2982_298230


namespace NUMINAMATH_CALUDE_sin_2phi_value_l2982_298261

theorem sin_2phi_value (φ : ℝ) (h : 7/13 + Real.sin φ = Real.cos φ) : 
  Real.sin (2 * φ) = 120/169 := by
  sorry

end NUMINAMATH_CALUDE_sin_2phi_value_l2982_298261


namespace NUMINAMATH_CALUDE_delta_y_over_delta_x_l2982_298264

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^2 - 4

-- Define the theorem
theorem delta_y_over_delta_x (Δx : ℝ) (Δy : ℝ) (h1 : f 1 = -2) (h2 : f (1 + Δx) = -2 + Δy) :
  Δy / Δx = 4 + 2 * Δx := by
  sorry

end NUMINAMATH_CALUDE_delta_y_over_delta_x_l2982_298264


namespace NUMINAMATH_CALUDE_total_short_trees_after_planting_l2982_298212

/-- Represents the types of trees in the park -/
inductive TreeType
  | Oak
  | Maple
  | Pine

/-- Represents the current state of trees in the park -/
structure ParkTrees where
  shortOak : ℕ
  shortMaple : ℕ
  shortPine : ℕ
  tallOak : ℕ
  tallMaple : ℕ
  tallPine : ℕ

/-- Calculates the new number of short trees after planting -/
def newShortTrees (park : ParkTrees) : ℕ :=
  let newOak := park.shortOak + 57
  let newMaple := park.shortMaple + (park.shortMaple * 3 / 10)  -- 30% increase
  let newPine := park.shortPine + (park.shortPine / 3)  -- 1/3 increase
  newOak + newMaple + newPine

/-- Theorem stating that the total number of short trees after planting is 153 -/
theorem total_short_trees_after_planting (park : ParkTrees) 
  (h1 : park.shortOak = 41)
  (h2 : park.shortMaple = 18)
  (h3 : park.shortPine = 24)
  (h4 : park.tallOak = 44)
  (h5 : park.tallMaple = 37)
  (h6 : park.tallPine = 17) :
  newShortTrees park = 153 := by
  sorry

end NUMINAMATH_CALUDE_total_short_trees_after_planting_l2982_298212


namespace NUMINAMATH_CALUDE_union_M_N_l2982_298238

def M : Set ℝ := {x | x ≥ -1}
def N : Set ℝ := {x | 2 - x^2 ≥ 0}

theorem union_M_N : M ∪ N = {x : ℝ | x ≥ -Real.sqrt 2} := by sorry

end NUMINAMATH_CALUDE_union_M_N_l2982_298238


namespace NUMINAMATH_CALUDE_impossibility_of_option_d_l2982_298211

-- Define the basic rhombus shape
structure Rhombus :=
  (color : Bool)  -- True for white, False for gray

-- Define the operation of rotation
def rotate (r : Rhombus) : Rhombus := r

-- Define a larger shape as a collection of rhombuses
def LargerShape := List Rhombus

-- Define the four options
def option_a : LargerShape := sorry
def option_b : LargerShape := sorry
def option_c : LargerShape := sorry
def option_d : LargerShape := sorry

-- Define a function to check if a larger shape can be constructed
def can_construct (shape : LargerShape) : Prop := sorry

-- State the theorem
theorem impossibility_of_option_d :
  can_construct option_a ∧
  can_construct option_b ∧
  can_construct option_c ∧
  ¬ can_construct option_d :=
sorry

end NUMINAMATH_CALUDE_impossibility_of_option_d_l2982_298211


namespace NUMINAMATH_CALUDE_ellipse_axis_distance_l2982_298241

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := 4 * (x - 2)^2 + 16 * y^2 = 64

-- Define the center of the ellipse
def center : ℝ × ℝ := (2, 0)

-- Define the major axis endpoint
def major_endpoint (C : ℝ × ℝ) : Prop :=
  ellipse C.1 C.2 ∧ (C.1 - center.1)^2 ≥ (C.2 - center.2)^2

-- Define the minor axis endpoint
def minor_endpoint (D : ℝ × ℝ) : Prop :=
  ellipse D.1 D.2 ∧ (D.1 - center.1)^2 < (D.2 - center.2)^2

-- Theorem statement
theorem ellipse_axis_distance (C D : ℝ × ℝ) :
  major_endpoint C → minor_endpoint D →
  (C.1 - D.1)^2 + (C.2 - D.2)^2 = 20 := by sorry

end NUMINAMATH_CALUDE_ellipse_axis_distance_l2982_298241


namespace NUMINAMATH_CALUDE_rectangle_circle_area_ratio_l2982_298207

theorem rectangle_circle_area_ratio (w l r : ℝ) (h1 : l = 2 * w) (h2 : 2 * l + 2 * w = 2 * Real.pi * r) :
  (l * w) / (Real.pi * r^2) = 2 * Real.pi / 9 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_circle_area_ratio_l2982_298207


namespace NUMINAMATH_CALUDE_periodic_function_value_l2982_298208

/-- Given a function f(x) = a*sin(πx + α) + b*cos(πx + β), if f(3) = 3, then f(2016) = -3 -/
theorem periodic_function_value (a b α β : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β)
  f 3 = 3 → f 2016 = -3 := by
  sorry

end NUMINAMATH_CALUDE_periodic_function_value_l2982_298208


namespace NUMINAMATH_CALUDE_two_digit_number_property_l2982_298291

theorem two_digit_number_property (a b : ℕ) : 
  b = 2 * a →
  10 * a + b - (10 * b + a) = 36 →
  (a + b) - (b - a) = 8 := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_property_l2982_298291


namespace NUMINAMATH_CALUDE_smallest_valid_number_l2982_298268

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧  -- four-digit number
  (let a := n / 1000
   let b := (n / 100) % 10
   let c := (n / 10) % 10
   let d := n % 10
   1000 * c + 100 * d + 10 * a + b - n = 5940) ∧  -- swapping condition
  n % 9 = 8 ∧  -- divisibility condition
  n % 2 = 1  -- odd number

theorem smallest_valid_number :
  is_valid_number 1979 ∧ ∀ m : ℕ, is_valid_number m → m ≥ 1979 :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_number_l2982_298268


namespace NUMINAMATH_CALUDE_absolute_value_and_trig_calculation_l2982_298225

theorem absolute_value_and_trig_calculation : |(-3 : ℝ)| + 2⁻¹ - Real.cos (π / 3) = 3 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_and_trig_calculation_l2982_298225


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_five_seven_l2982_298259

/-- Represents a repeating decimal with a single digit repeating infinitely after the decimal point. -/
def RepeatingDecimal (n : ℕ) : ℚ := n / 9

theorem sum_of_repeating_decimals_five_seven :
  RepeatingDecimal 5 + RepeatingDecimal 7 = 4 / 3 := by sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_five_seven_l2982_298259


namespace NUMINAMATH_CALUDE_composition_equals_26_l2982_298293

-- Define the functions f and g
def f (x : ℝ) : ℝ := x + 3
def g (x : ℝ) : ℝ := 2 * x

-- Define the inverse functions
noncomputable def f_inv (x : ℝ) : ℝ := x - 3
noncomputable def g_inv (x : ℝ) : ℝ := x / 2

-- State the theorem
theorem composition_equals_26 : f (g_inv (f_inv (f_inv (g (f 23))))) = 26 := by sorry

end NUMINAMATH_CALUDE_composition_equals_26_l2982_298293


namespace NUMINAMATH_CALUDE_inequality_proof_l2982_298281

theorem inequality_proof (x y z : ℝ) 
  (sum_zero : x + y + z = 0) 
  (abs_sum_le_one : |x| + |y| + |z| ≤ 1) : 
  x + y/2 + z/3 ≤ 1/3 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2982_298281


namespace NUMINAMATH_CALUDE_shortest_chord_through_A_equals_4_l2982_298246

-- Define the circle M
def circle_M (x y : ℝ) : Prop := x^2 + y^2 - 6*x = 0

-- Define point A
def point_A : ℝ × ℝ := (1, 1)

-- Define the function to calculate the shortest chord length
noncomputable def shortest_chord_length (c : (ℝ → ℝ → Prop)) (p : ℝ × ℝ) : ℝ :=
  sorry -- Implementation details are omitted

-- Theorem statement
theorem shortest_chord_through_A_equals_4 :
  shortest_chord_length circle_M point_A = 4 := by sorry

end NUMINAMATH_CALUDE_shortest_chord_through_A_equals_4_l2982_298246


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l2982_298224

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7, 8, 9}
def A : Set Nat := {3, 4, 5, 6}
def B : Set Nat := {5, 6, 7, 8, 9}

theorem complement_intersection_theorem :
  (Aᶜ ∩ B) = {7, 8, 9} :=
by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l2982_298224


namespace NUMINAMATH_CALUDE_solution_of_linear_equation_l2982_298220

theorem solution_of_linear_equation (a : ℚ) : 
  (∃ x y : ℚ, x = 2 ∧ y = 2 ∧ a * x + y = 5) → a = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_solution_of_linear_equation_l2982_298220
