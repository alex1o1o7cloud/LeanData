import Mathlib

namespace NUMINAMATH_CALUDE_glasses_wearers_properties_l995_99598

-- Define the universe of women
variable (Woman : Type)

-- Define predicates
variable (wears_glasses : Woman → Prop)
variable (knows_english : Woman → Prop)
variable (wears_chignon : Woman → Prop)
variable (has_seven_children : Woman → Prop)

-- Define the five statements as axioms
axiom statement1 : ∀ w : Woman, has_seven_children w → knows_english w → wears_chignon w
axiom statement2 : ∀ w : Woman, wears_glasses w → (has_seven_children w ∨ knows_english w)
axiom statement3 : ∀ w : Woman, ¬has_seven_children w → wears_glasses w → wears_chignon w
axiom statement4 : ∀ w : Woman, has_seven_children w → wears_glasses w → knows_english w
axiom statement5 : ∀ w : Woman, wears_chignon w → ¬has_seven_children w

-- Theorem to prove
theorem glasses_wearers_properties :
  ∀ w : Woman, wears_glasses w → (knows_english w ∧ wears_chignon w ∧ ¬has_seven_children w) := by
  sorry

end NUMINAMATH_CALUDE_glasses_wearers_properties_l995_99598


namespace NUMINAMATH_CALUDE_complex_product_magnitude_l995_99526

theorem complex_product_magnitude (z₁ z₂ : ℂ) (h₁ : z₁ = 1 + 2*I) (h₂ : z₂ = 1 - I) : 
  Complex.abs (z₁ * z₂) = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_magnitude_l995_99526


namespace NUMINAMATH_CALUDE_inequality_solution_set_l995_99505

-- Define the inequality
def inequality (x : ℝ) : Prop := (x - 1) * (x - 2) ≤ 0

-- Define the solution set
def solution_set : Set ℝ := {x | 1 ≤ x ∧ x ≤ 2}

-- Theorem stating that the solution set of the inequality is [1, 2]
theorem inequality_solution_set : 
  {x : ℝ | inequality x} = solution_set := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l995_99505


namespace NUMINAMATH_CALUDE_inequality_proof_l995_99528

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : 2 * a^2 + b^2 = 9 * c^2) : 
  (2 * c / a) + (c / b) ≥ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l995_99528


namespace NUMINAMATH_CALUDE_sector_area_l995_99553

/-- The area of a sector with perimeter 1 and central angle 1 radian is 1/18 -/
theorem sector_area (r : ℝ) (l : ℝ) (h1 : l + 2*r = 1) (h2 : l = r) : r^2/2 = 1/18 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l995_99553


namespace NUMINAMATH_CALUDE_closest_point_l995_99512

def v (t : ℝ) : ℝ × ℝ × ℝ := (3 - 4*t, -1 + 3*t, 2 + 5*t)

def a : ℝ × ℝ × ℝ := (-2, 7, 0)

def direction : ℝ × ℝ × ℝ := (-4, 3, 5)

theorem closest_point (t : ℝ) : 
  (v t - a) • direction = 0 ↔ t = 17/25 := by sorry

end NUMINAMATH_CALUDE_closest_point_l995_99512


namespace NUMINAMATH_CALUDE_first_half_speed_calculation_l995_99589

/-- Represents a trip with two halves -/
structure Trip where
  total_distance : ℝ
  first_half_speed : ℝ
  second_half_time_multiplier : ℝ
  average_speed : ℝ

/-- Calculates the speed of the first half of the trip -/
def first_half_speed (t : Trip) : ℝ :=
  2 * t.average_speed

/-- Theorem stating the conditions and the result to be proved -/
theorem first_half_speed_calculation (t : Trip)
  (h1 : t.total_distance = 640)
  (h2 : t.second_half_time_multiplier = 3)
  (h3 : t.average_speed = 40) :
  first_half_speed t = 80 := by
  sorry

#eval first_half_speed { total_distance := 640, first_half_speed := 0, second_half_time_multiplier := 3, average_speed := 40 }

end NUMINAMATH_CALUDE_first_half_speed_calculation_l995_99589


namespace NUMINAMATH_CALUDE_oranges_apples_balance_l995_99585

/-- Given that 9 oranges have the same weight as 6 apples, 
    this function calculates the number of apples that would 
    balance the weight of a given number of oranges. -/
def applesEquivalent (numOranges : ℕ) : ℕ :=
  (2 * numOranges) / 3

/-- Theorem stating that 30 apples balance the weight of 45 oranges, 
    given the weight ratio between oranges and apples. -/
theorem oranges_apples_balance :
  applesEquivalent 45 = 30 := by
  sorry

end NUMINAMATH_CALUDE_oranges_apples_balance_l995_99585


namespace NUMINAMATH_CALUDE_golf_over_par_l995_99544

/-- Calculates the number of strokes over par for a golfer --/
def strokes_over_par (holes : ℕ) (avg_strokes_per_hole : ℕ) (par_per_hole : ℕ) : ℕ :=
  (holes * avg_strokes_per_hole) - (holes * par_per_hole)

/-- Theorem stating that a golfer playing 9 holes with an average of 4 strokes per hole
    and a par of 3 per hole will be 9 strokes over par --/
theorem golf_over_par :
  strokes_over_par 9 4 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_golf_over_par_l995_99544


namespace NUMINAMATH_CALUDE_canada_human_beaver_ratio_l995_99580

/-- The ratio of humans to beavers in Canada -/
def human_beaver_ratio (moose_population : ℕ) (human_population : ℕ) : ℚ :=
  human_population / (2 * moose_population)

/-- Theorem stating the ratio of humans to beavers in Canada -/
theorem canada_human_beaver_ratio :
  human_beaver_ratio 1000000 38000000 = 19 := by
  sorry

end NUMINAMATH_CALUDE_canada_human_beaver_ratio_l995_99580


namespace NUMINAMATH_CALUDE_distributive_property_first_calculation_l995_99534

theorem distributive_property (a b c : ℤ) : a * (b + c) = a * b + a * c := by sorry

theorem first_calculation : -17 * 43 + (-17) * 20 - (-17) * 163 = 1700 := by sorry

end NUMINAMATH_CALUDE_distributive_property_first_calculation_l995_99534


namespace NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l995_99537

theorem consecutive_odd_integers_sum (a : ℤ) : 
  (∃ (b c : ℤ), b = a + 2 ∧ c = a + 4 ∧ a + c = 100) →
  a + (a + 2) + (a + 4) = 150 := by
sorry

end NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l995_99537


namespace NUMINAMATH_CALUDE_number_line_order_l995_99588

theorem number_line_order (x y : ℝ) : x > y ↔ (∃ (d : ℝ), d > 0 ∧ x = y + d) :=
sorry

end NUMINAMATH_CALUDE_number_line_order_l995_99588


namespace NUMINAMATH_CALUDE_problem_solution_l995_99581

def A (a : ℕ) : Set ℝ := {x : ℝ | |x + 1| > a}

theorem problem_solution (a : ℕ) 
  (h1 : 3/2 ∈ A a) 
  (h2 : 1/2 ∉ A a) 
  (h3 : a > 0) :
  (a = 2) ∧ 
  (∀ m n s : ℝ, m > 0 → n > 0 → s > 0 → m + n + Real.sqrt 2 * s = a → 
    m^2 + n^2 + s^2 ≥ 1 ∧ ∃ m₀ n₀ s₀, m₀ > 0 ∧ n₀ > 0 ∧ s₀ > 0 ∧ 
      m₀ + n₀ + Real.sqrt 2 * s₀ = a ∧ m₀^2 + n₀^2 + s₀^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l995_99581


namespace NUMINAMATH_CALUDE_smallest_solution_l995_99513

/-- Reverses the digits of a positive integer -/
def reverseDigits (n : ℕ) : ℕ := sorry

/-- Checks if a number satisfies the condition K = 9F -/
def satisfiesCondition (k : ℕ) : Prop :=
  k = 9 * (reverseDigits k)

theorem smallest_solution :
  satisfiesCondition 9801 ∧ ∀ k < 9801, ¬satisfiesCondition k := by sorry

end NUMINAMATH_CALUDE_smallest_solution_l995_99513


namespace NUMINAMATH_CALUDE_five_teachers_three_types_l995_99570

/-- The number of ways to assign teachers to question types. -/
def assignment_count (teachers : ℕ) (question_types : ℕ) : ℕ :=
  -- Number of ways to assign teachers to question types
  -- with at least one teacher per type
  sorry

/-- Theorem stating the number of assignment methods for 5 teachers and 3 question types. -/
theorem five_teachers_three_types : assignment_count 5 3 = 150 := by
  sorry

end NUMINAMATH_CALUDE_five_teachers_three_types_l995_99570


namespace NUMINAMATH_CALUDE_intersection_eq_A_intersection_nonempty_l995_99539

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | 2*a < x ∧ x < a + 1}
def B : Set ℝ := {x | x < -1 ∨ x > 3}

-- Theorem for the first question
theorem intersection_eq_A (a : ℝ) : A a ∩ B = A a ↔ a ≤ -2 ∨ a ≥ 1 := by sorry

-- Theorem for the second question
theorem intersection_nonempty (a : ℝ) : (A a ∩ B).Nonempty ↔ a < -1/2 := by sorry

end NUMINAMATH_CALUDE_intersection_eq_A_intersection_nonempty_l995_99539


namespace NUMINAMATH_CALUDE_sports_club_overlap_l995_99545

theorem sports_club_overlap (N B T Neither : ℕ) (h1 : N = 30) (h2 : B = 17) (h3 : T = 19) (h4 : Neither = 2) :
  B + T - N + Neither = 8 := by
  sorry

end NUMINAMATH_CALUDE_sports_club_overlap_l995_99545


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l995_99522

theorem solution_set_equivalence (x : ℝ) : (x + 3) * (x - 5) < 0 ↔ -3 < x ∧ x < 5 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l995_99522


namespace NUMINAMATH_CALUDE_store_pricing_l995_99562

/-- The price of a chair in dollars -/
def chair_price : ℝ := 60 - 52.5

/-- The price of a table in dollars -/
def table_price : ℝ := 52.5

/-- The price of 2 chairs and 1 table in dollars -/
def two_chairs_one_table : ℝ := 2 * chair_price + table_price

/-- The price of 1 chair and 2 tables in dollars -/
def one_chair_two_tables : ℝ := chair_price + 2 * table_price

theorem store_pricing :
  two_chairs_one_table / one_chair_two_tables = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_store_pricing_l995_99562


namespace NUMINAMATH_CALUDE_solve_for_k_l995_99551

/-- A function that represents the linearity condition of the equation -/
def is_linear (k : ℤ) : Prop := ∀ x y : ℝ, ∃ a b c : ℝ, 2 * x^(|k|) + (k - 1) * y = a * x + b * y + c

/-- The main theorem stating the conditions and the conclusion about the value of k -/
theorem solve_for_k (k : ℤ) (h1 : is_linear k) (h2 : k - 1 ≠ 0) : k = -1 := by
  sorry


end NUMINAMATH_CALUDE_solve_for_k_l995_99551


namespace NUMINAMATH_CALUDE_remainder_of_power_division_l995_99571

theorem remainder_of_power_division (n : ℕ) : 
  (2^160 + 160) % (2^81 + 2^41 + 1) = 159 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_power_division_l995_99571


namespace NUMINAMATH_CALUDE_bijection_ordered_images_l995_99576

theorem bijection_ordered_images 
  (f : ℕ → ℕ) (hf : Function.Bijective f) : 
  ∃ (a d : ℕ), 
    0 < a ∧ 0 < d ∧ 
    a < a + d ∧ a + d < a + 2*d ∧
    f a < f (a + d) ∧ f (a + d) < f (a + 2*d) :=
sorry

end NUMINAMATH_CALUDE_bijection_ordered_images_l995_99576


namespace NUMINAMATH_CALUDE_zeros_product_greater_than_e_l995_99575

open Real

theorem zeros_product_greater_than_e (a : ℝ) (x₁ x₂ : ℝ) :
  x₁ > 0 → x₂ > 0 →
  (log x₁ = a * x₁^2) →
  (log x₂ = a * x₂^2) →
  x₁ * x₂ > Real.exp 1 := by
sorry

end NUMINAMATH_CALUDE_zeros_product_greater_than_e_l995_99575


namespace NUMINAMATH_CALUDE_distance_traveled_l995_99559

-- Define the average speed of car R
def speed_R : ℝ := 34.05124837953327

-- Define the time difference between car R and car P
def time_difference : ℝ := 2

-- Define the speed difference between car P and car R
def speed_difference : ℝ := 10

-- Theorem statement
theorem distance_traveled (t : ℝ) (h : t > time_difference) :
  speed_R * t = (speed_R + speed_difference) * (t - time_difference) →
  speed_R * t = 300 :=
by sorry

end NUMINAMATH_CALUDE_distance_traveled_l995_99559


namespace NUMINAMATH_CALUDE_negative_two_inequality_l995_99563

theorem negative_two_inequality (a b : ℝ) (h : a > b) : -2 * a < -2 * b := by
  sorry

end NUMINAMATH_CALUDE_negative_two_inequality_l995_99563


namespace NUMINAMATH_CALUDE_degree_not_determined_by_A_P_l995_99523

/-- A characteristic associated with a polynomial -/
def A_P (P : Polynomial ℝ) : Type :=
  sorry

/-- Theorem stating that the degree of a polynomial cannot be uniquely determined from A_P -/
theorem degree_not_determined_by_A_P :
  ∃ (P₁ P₂ : Polynomial ℝ), A_P P₁ = A_P P₂ ∧ Polynomial.degree P₁ ≠ Polynomial.degree P₂ :=
sorry

end NUMINAMATH_CALUDE_degree_not_determined_by_A_P_l995_99523


namespace NUMINAMATH_CALUDE_n_pointed_star_angle_sum_l995_99540

/-- A structure representing an n-pointed star formed from a convex polygon. -/
structure NPointedStar where
  n : ℕ
  n_ge_5 : n ≥ 5

/-- The sum of interior angles at the n points of an n-pointed star. -/
def interior_angle_sum (star : NPointedStar) : ℝ :=
  180 * (star.n - 4)

/-- Theorem stating that the sum of interior angles of an n-pointed star is 180(n-4) degrees. -/
theorem n_pointed_star_angle_sum (star : NPointedStar) :
  interior_angle_sum star = 180 * (star.n - 4) := by
  sorry

end NUMINAMATH_CALUDE_n_pointed_star_angle_sum_l995_99540


namespace NUMINAMATH_CALUDE_carla_hits_nine_l995_99535

/-- Represents a player in the dart contest -/
inductive Player : Type
| Anne : Player
| Joe : Player
| Carla : Player
| Larry : Player
| Naomi : Player
| Mike : Player

/-- The score of each player -/
def score (p : Player) : ℕ :=
  match p with
  | Player.Anne => 21
  | Player.Joe => 18
  | Player.Carla => 14
  | Player.Larry => 22
  | Player.Naomi => 25
  | Player.Mike => 13

/-- The set of possible scores for each throw -/
def possible_scores : Set ℕ := Finset.range 15

/-- Predicate to check if a list of scores is valid for a player -/
def valid_scores (s : List ℕ) (p : Player) : Prop :=
  s.length = 3 ∧ 
  s.sum = score p ∧
  s.toFinset.card = 3 ∧
  ∀ x ∈ s, x ∈ possible_scores

theorem carla_hits_nine : 
  ∃! (p : Player), ∃ (s : List ℕ), valid_scores s p ∧ 9 ∈ s ∧ p = Player.Carla :=
sorry

end NUMINAMATH_CALUDE_carla_hits_nine_l995_99535


namespace NUMINAMATH_CALUDE_range_of_a_l995_99525

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - a ≥ 0) ∧ 
  (∃ x₀ : ℝ, x₀^2 + 2*a*x₀ + 2 - a = 0) →
  (a ≤ -2 ∨ a = 1) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l995_99525


namespace NUMINAMATH_CALUDE_complex_product_real_l995_99567

theorem complex_product_real (a : ℝ) : 
  let z₁ : ℂ := 2 - I
  let z₂ : ℂ := a + 2*I
  (z₁ * z₂).im = 0 → a = 4 := by sorry

end NUMINAMATH_CALUDE_complex_product_real_l995_99567


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l995_99552

/-- 
Given an arithmetic sequence where:
- a is the first term
- d is the common difference
- The first term (a) is 5/6
- The seventeenth term (a + 16d) is 5/8

Prove that the ninth term (a + 8d) is 15/16
-/
theorem arithmetic_sequence_ninth_term 
  (a d : ℚ) 
  (h1 : a = 5/6) 
  (h2 : a + 16*d = 5/8) : 
  a + 8*d = 15/16 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l995_99552


namespace NUMINAMATH_CALUDE_f_equals_mod_l995_99560

def g (n : ℤ) : ℕ :=
  if n ≥ 1 then 1 else 0

def f : ℕ → ℕ → ℕ
  | 0, m => 0
  | n+1, m => ((1 - g m + g m * g (m-1-(f n m))) * (1 + f n m)) % m

theorem f_equals_mod (n m : ℕ) : f n m = n % m := by
  sorry

end NUMINAMATH_CALUDE_f_equals_mod_l995_99560


namespace NUMINAMATH_CALUDE_inequality_and_floor_function_l995_99510

theorem inequality_and_floor_function (n : ℕ) : 
  (Real.sqrt (n + 1) + 2 * Real.sqrt n < Real.sqrt (9 * n + 3)) ∧ 
  ¬(∃ n : ℕ, ⌊Real.sqrt (n + 1) + 2 * Real.sqrt n⌋ < ⌊Real.sqrt (9 * n + 3)⌋) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_floor_function_l995_99510


namespace NUMINAMATH_CALUDE_kitchen_renovation_time_percentage_l995_99554

theorem kitchen_renovation_time_percentage (
  bedroom_count : Nat) 
  (bedroom_time : Nat) 
  (total_time : Nat) 
  (kitchen_time : Nat) : 
  bedroom_count = 3 → 
  bedroom_time = 4 → 
  total_time = 54 → 
  kitchen_time = 6 → 
  (kitchen_time - bedroom_time) / bedroom_time * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_kitchen_renovation_time_percentage_l995_99554


namespace NUMINAMATH_CALUDE_reciprocal_inequality_l995_99506

theorem reciprocal_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) : 1/a > 1/b := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_inequality_l995_99506


namespace NUMINAMATH_CALUDE_min_coeff_x2_and_coeff_x7_l995_99565

/-- The function f(x) = (1+x)^m + (1+x)^n -/
def f (m n : ℕ+) (x : ℝ) : ℝ := (1 + x)^(m : ℕ) + (1 + x)^(n : ℕ)

theorem min_coeff_x2_and_coeff_x7 (m n : ℕ+) :
  (m : ℕ) + n = 19 →
  (∃ c : ℕ, c = Nat.choose m 1 + Nat.choose n 1 ∧ c = 19) →
  let coeff_x2 := Nat.choose m 2 + Nat.choose n 2
  ∃ (m' n' : ℕ+),
    (∀ k l : ℕ+, Nat.choose k 2 + Nat.choose l 2 ≥ coeff_x2) ∧
    coeff_x2 = 81 ∧
    Nat.choose m' 7 + Nat.choose n' 7 = 156 :=
sorry

end NUMINAMATH_CALUDE_min_coeff_x2_and_coeff_x7_l995_99565


namespace NUMINAMATH_CALUDE_marble_weight_proof_l995_99564

/-- The weight of each of the two identical pieces of marble -/
def x : ℝ := 0.335

/-- The weight of the third piece of marble -/
def third_piece_weight : ℝ := 0.08

/-- The total weight of all three pieces of marble -/
def total_weight : ℝ := 0.75

theorem marble_weight_proof :
  2 * x + third_piece_weight = total_weight :=
by sorry

end NUMINAMATH_CALUDE_marble_weight_proof_l995_99564


namespace NUMINAMATH_CALUDE_division_problem_l995_99577

theorem division_problem (smaller larger quotient : ℕ) : 
  larger - smaller = 1365 →
  larger = 1620 →
  larger = quotient * smaller + 15 →
  quotient = 6 :=
by sorry

end NUMINAMATH_CALUDE_division_problem_l995_99577


namespace NUMINAMATH_CALUDE_popped_kernel_probability_l995_99569

theorem popped_kernel_probability (total_kernels : ℝ) (h_total_positive : 0 < total_kernels) : 
  let white_kernels := (3 / 5) * total_kernels
  let yellow_kernels := (2 / 5) * total_kernels
  let popped_white := (2 / 5) * white_kernels
  let popped_yellow := (4 / 5) * yellow_kernels
  let total_popped := popped_white + popped_yellow
  (popped_white / total_popped) = (3 / 7) :=
by sorry

end NUMINAMATH_CALUDE_popped_kernel_probability_l995_99569


namespace NUMINAMATH_CALUDE_range_of_a_l995_99590

theorem range_of_a (a : ℝ) : 
  let A := {x : ℝ | x ≤ a}
  let B := Set.Iio 2
  A ⊆ B → a < 2 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l995_99590


namespace NUMINAMATH_CALUDE_z_in_third_quadrant_l995_99502

def complex_number_quadrant (z : ℂ) : Prop :=
  Real.sign z.re = -1 ∧ Real.sign z.im = -1

theorem z_in_third_quadrant :
  let z : ℂ := (-2 - Complex.I) * (3 + Complex.I)
  complex_number_quadrant z := by
  sorry

end NUMINAMATH_CALUDE_z_in_third_quadrant_l995_99502


namespace NUMINAMATH_CALUDE_four_type_B_in_rewards_l995_99587

/-- Represents the cost and purchasing details of appliances A and B -/
structure AppliancePurchase where
  cost_A : ℕ  -- Cost of appliance A in yuan
  cost_B : ℕ  -- Cost of appliance B in yuan
  total_units : ℕ  -- Total units to purchase
  max_amount : ℕ  -- Maximum amount to spend in yuan
  max_A : ℕ  -- Maximum units of appliance A
  sell_A : ℕ  -- Selling price of appliance A in yuan
  sell_B : ℕ  -- Selling price of appliance B in yuan
  reward_units : ℕ  -- Units taken out for employee rewards
  profit : ℕ  -- Profit after selling remaining appliances in yuan

/-- Theorem stating that given the conditions, 4 units of type B are among the 10 reward units -/
theorem four_type_B_in_rewards (p : AppliancePurchase) 
  (h1 : p.cost_B = p.cost_A + 100)
  (h2 : 10000 / p.cost_A = 12000 / p.cost_B)
  (h3 : p.total_units = 100)
  (h4 : p.cost_A * 67 + p.cost_B * 33 ≤ p.max_amount)
  (h5 : p.max_A = 67)
  (h6 : p.sell_A = 600)
  (h7 : p.sell_B = 750)
  (h8 : p.reward_units = 10)
  (h9 : p.profit = 5050) :
  ∃ (a b : ℕ), a + b = p.reward_units ∧ b = 4 := by sorry

end NUMINAMATH_CALUDE_four_type_B_in_rewards_l995_99587


namespace NUMINAMATH_CALUDE_midpoint_theorem_l995_99584

/-- Given points A, B, and C in a 2D plane, where B is the midpoint of AC -/
structure Midpoint2D where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The midpoint condition for 2D points -/
def isMidpoint (m : Midpoint2D) : Prop :=
  m.B.1 = (m.A.1 + m.C.1) / 2 ∧ m.B.2 = (m.A.2 + m.C.2) / 2

/-- Theorem: If B(3,4) is the midpoint of AC where A(1,1), then C is (5,7) -/
theorem midpoint_theorem (m : Midpoint2D) 
    (h1 : m.A = (1, 1))
    (h2 : m.B = (3, 4))
    (h3 : isMidpoint m) :
    m.C = (5, 7) := by
  sorry


end NUMINAMATH_CALUDE_midpoint_theorem_l995_99584


namespace NUMINAMATH_CALUDE_point_movement_l995_99566

/-- Given a point P(-5, -2) in the Cartesian coordinate system, 
    moving it 3 units left and 2 units up results in the point (-8, 0). -/
theorem point_movement : 
  let P : ℝ × ℝ := (-5, -2)
  let left_movement : ℝ := 3
  let up_movement : ℝ := 2
  let new_x : ℝ := P.1 - left_movement
  let new_y : ℝ := P.2 + up_movement
  (new_x, new_y) = (-8, 0) := by sorry

end NUMINAMATH_CALUDE_point_movement_l995_99566


namespace NUMINAMATH_CALUDE_tangent_line_to_parabola_l995_99515

/-- The equation of the tangent line to the parabola y = x^2 that is parallel to the line 2x - y + 4 = 0 is 2x - y - 1 = 0 -/
theorem tangent_line_to_parabola : 
  ∀ (x y : ℝ), 
  (y = x^2) →  -- Parabola equation
  (∃ (k : ℝ), k * (2 * x - y + 4) = 0) →  -- Parallel condition
  (2 * x - y - 1 = 0) -- Tangent line equation
  := by sorry

end NUMINAMATH_CALUDE_tangent_line_to_parabola_l995_99515


namespace NUMINAMATH_CALUDE_max_k_for_cube_sum_inequality_l995_99599

theorem max_k_for_cube_sum_inequality (m n : ℕ+) (h : m^3 + n^3 > (m + n)^2) :
  (∃ k : ℕ+, ∀ j : ℕ+, (m^3 + n^3 ≥ (m + n)^2 + j) → j ≤ k) ∧
  (∀ k : ℕ+, (∀ m n : ℕ+, m^3 + n^3 > (m + n)^2 → m^3 + n^3 ≥ (m + n)^2 + k) → k ≤ 10) :=
by sorry

end NUMINAMATH_CALUDE_max_k_for_cube_sum_inequality_l995_99599


namespace NUMINAMATH_CALUDE_two_integer_segments_l995_99592

/-- A right triangle with integer leg lengths 24 and 25 -/
structure RightTriangle where
  /-- Length of the first leg -/
  de : ℕ
  /-- Length of the second leg -/
  ef : ℕ
  /-- Assumption that the first leg has length 24 -/
  de_eq : de = 24
  /-- Assumption that the second leg has length 25 -/
  ef_eq : ef = 25

/-- The number of integer-length line segments from vertex E to hypotenuse DF -/
def count_integer_segments (t : RightTriangle) : ℕ :=
  2

/-- Theorem stating that there are exactly 2 integer-length line segments 
    from vertex E to hypotenuse DF in the given right triangle -/
theorem two_integer_segments (t : RightTriangle) :
  count_integer_segments t = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_integer_segments_l995_99592


namespace NUMINAMATH_CALUDE_lost_revenue_example_l995_99568

/-- Represents a movie theater with its capacity, ticket price, and sold tickets. -/
structure MovieTheater where
  capacity : Nat
  ticketPrice : Nat
  soldTickets : Nat

/-- Calculates the lost revenue for a movie theater. -/
def lostRevenue (theater : MovieTheater) : Nat :=
  theater.capacity * theater.ticketPrice - theater.soldTickets * theater.ticketPrice

/-- Theorem: The lost revenue for the given theater is $208.00. -/
theorem lost_revenue_example :
  let theater : MovieTheater := {
    capacity := 50,
    ticketPrice := 8,
    soldTickets := 24
  }
  lostRevenue theater = 208 := by
  sorry


end NUMINAMATH_CALUDE_lost_revenue_example_l995_99568


namespace NUMINAMATH_CALUDE_air_conditioner_sales_theorem_l995_99582

/-- Represents the shopping mall's air conditioner purchases and sales --/
structure AirConditionerSales where
  first_purchase_total : ℝ
  first_purchase_unit_price : ℝ
  second_purchase_total : ℝ
  second_purchase_unit_price : ℝ
  selling_price_increase : ℝ
  profit_rate : ℝ
  discount_rate : ℝ

/-- Theorem about the unit cost of the first purchase and maximum discounted units --/
theorem air_conditioner_sales_theorem (sale : AirConditionerSales)
  (h1 : sale.first_purchase_total = 24000)
  (h2 : sale.first_purchase_unit_price = 3000)
  (h3 : sale.second_purchase_total = 52000)
  (h4 : sale.second_purchase_unit_price = sale.first_purchase_unit_price + 200)
  (h5 : sale.selling_price_increase = 200)
  (h6 : sale.profit_rate = 0.22)
  (h7 : sale.discount_rate = 0.95) :
  ∃ (first_unit_cost max_discounted : ℝ),
    first_unit_cost = 2400 ∧
    max_discounted = 8 ∧
    (sale.first_purchase_total / first_unit_cost) * 2 = sale.second_purchase_total / sale.second_purchase_unit_price ∧
    sale.first_purchase_unit_price * (sale.first_purchase_total / first_unit_cost) +
    (sale.first_purchase_unit_price + sale.selling_price_increase) * sale.discount_rate * max_discounted +
    (sale.first_purchase_unit_price + sale.selling_price_increase) * ((sale.second_purchase_total / sale.second_purchase_unit_price) - max_discounted) ≥
    (sale.first_purchase_total + sale.second_purchase_total) * (1 + sale.profit_rate) :=
by sorry

end NUMINAMATH_CALUDE_air_conditioner_sales_theorem_l995_99582


namespace NUMINAMATH_CALUDE_pythagoras_field_planted_fraction_l995_99507

theorem pythagoras_field_planted_fraction :
  ∀ (a b c x : ℝ),
    a = 5 ∧ b = 12 ∧ c^2 = a^2 + b^2 →
    x > 0 →
    (a - x) * (b - x) / 2 = 1 →
    (a * b / 2 - x^2) / (a * b / 2) = 2951 / 3000 := by
  sorry

end NUMINAMATH_CALUDE_pythagoras_field_planted_fraction_l995_99507


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l995_99591

/-- An arithmetic sequence with given properties -/
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  a 5 = 11 ∧ a 12 = 31 ∧ ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

theorem arithmetic_sequence_properties (a : ℕ → ℤ) 
  (h : arithmetic_sequence a) : 
  a 1 = -2 ∧ 
  (∀ n : ℕ, a (n + 1) - a n = 3) ∧
  a 20 = 55 ∧
  (∀ n : ℕ, a n = 3 * n - 5) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l995_99591


namespace NUMINAMATH_CALUDE_lock_cost_l995_99533

def total_cost : ℝ := 360

theorem lock_cost (helmet : ℝ) (bicycle : ℝ) (lock : ℝ) : 
  bicycle = 5 * helmet → 
  lock = helmet / 2 → 
  bicycle + helmet + lock = total_cost → 
  lock = 27.72 := by sorry

end NUMINAMATH_CALUDE_lock_cost_l995_99533


namespace NUMINAMATH_CALUDE_commute_speed_l995_99532

theorem commute_speed (v : ℝ) (h1 : v > 0) : 
  (18 / v + 18 / 30 = 1) → v = 45 := by
  sorry

end NUMINAMATH_CALUDE_commute_speed_l995_99532


namespace NUMINAMATH_CALUDE_store_earnings_calculation_l995_99596

/-- Represents the earnings from selling bottled drinks in a country store. -/
def storeEarnings (colaCost juiceCost waterCost : ℝ) (colaSold juiceSold waterSold : ℕ) : ℝ :=
  colaCost * colaSold + juiceCost * juiceSold + waterCost * waterSold

/-- Theorem stating that the store's earnings from selling the specified quantities of drinks at given prices is $88. -/
theorem store_earnings_calculation :
  storeEarnings 3 1.5 1 15 12 25 = 88 := by
  sorry

end NUMINAMATH_CALUDE_store_earnings_calculation_l995_99596


namespace NUMINAMATH_CALUDE_final_positions_l995_99555

/-- Represents the positions of the cat -/
inductive CatPosition
| TopLeft
| TopRight
| BottomRight
| BottomLeft

/-- Represents the positions of the mouse -/
inductive MousePosition
| TopLeft
| TopMiddleRight
| MiddleRight
| BottomRight
| BottomMiddleLeft
| MiddleLeft

/-- The number of squares in the cat's cycle -/
def catCycleLength : Nat := 4

/-- The number of segments in the mouse's cycle -/
def mouseCycleLength : Nat := 6

/-- The total number of moves -/
def totalMoves : Nat := 320

/-- Function to determine the cat's position after a given number of moves -/
def catPosition (moves : Nat) : CatPosition := 
  match moves % catCycleLength with
  | 0 => CatPosition.TopLeft
  | 1 => CatPosition.TopRight
  | 2 => CatPosition.BottomRight
  | _ => CatPosition.BottomLeft

/-- Function to determine the mouse's position after a given number of moves -/
def mousePosition (moves : Nat) : MousePosition := 
  match moves % mouseCycleLength with
  | 0 => MousePosition.TopLeft
  | 1 => MousePosition.TopMiddleRight
  | 2 => MousePosition.MiddleRight
  | 3 => MousePosition.BottomRight
  | 4 => MousePosition.BottomMiddleLeft
  | _ => MousePosition.MiddleLeft

theorem final_positions : 
  catPosition totalMoves = CatPosition.TopLeft ∧ 
  mousePosition totalMoves = MousePosition.MiddleRight := by
  sorry

end NUMINAMATH_CALUDE_final_positions_l995_99555


namespace NUMINAMATH_CALUDE_middle_segment_length_l995_99524

theorem middle_segment_length (a b c : ℝ) (ha : a = 1) (hb : b = 3) 
  (hc : c * c = a * b) : c = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_middle_segment_length_l995_99524


namespace NUMINAMATH_CALUDE_product_of_terms_l995_99557

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem product_of_terms (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 5) ^ 2 - 4 * (a 5) + 3 = 0 →
  (a 7) ^ 2 - 4 * (a 7) + 3 = 0 →
  a 2 * a 10 = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_product_of_terms_l995_99557


namespace NUMINAMATH_CALUDE_product_abcd_l995_99511

/-- Given a system of equations, prove that the product of a, b, c, and d is equal to 58653 / 10716361 -/
theorem product_abcd (a b c d : ℚ) : 
  (4*a + 5*b + 7*c + 9*d = 56) →
  (4*(d+c) = b) →
  (4*b + 2*c = a) →
  (c - 2 = d) →
  a * b * c * d = 58653 / 10716361 := by
  sorry

end NUMINAMATH_CALUDE_product_abcd_l995_99511


namespace NUMINAMATH_CALUDE_sqrt_three_times_sqrt_six_l995_99593

theorem sqrt_three_times_sqrt_six : Real.sqrt 3 * Real.sqrt 6 = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_times_sqrt_six_l995_99593


namespace NUMINAMATH_CALUDE_stratified_sample_sum_l995_99501

/-- Represents the number of items in each category -/
def categories : List ℕ := [40, 10, 30, 20]

/-- Total number of items -/
def total : ℕ := categories.sum

/-- Sample size -/
def sample_size : ℕ := 20

/-- Calculates the number of items sampled from a category -/
def sampled_items (category_size : ℕ) : ℕ :=
  (category_size * sample_size) / total

/-- Theorem stating that the sum of sampled items from categories with 10 and 20 items is 6 -/
theorem stratified_sample_sum :
  sampled_items (categories[1]) + sampled_items (categories[3]) = 6 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_sum_l995_99501


namespace NUMINAMATH_CALUDE_problem_proof_l995_99538

theorem problem_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 2*b = 1) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + 2*y = 1 ∧ x*y > a*b) → a*b ≤ 1/8 ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → x + 2*y = 1 → 1/x + 8/y ≥ 25) ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → x + 2*y = 1 → x^2 + 4*y^2 ≥ 1/2) :=
by sorry

end NUMINAMATH_CALUDE_problem_proof_l995_99538


namespace NUMINAMATH_CALUDE_geometric_sequence_a8_l995_99597

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_a8 (a : ℕ → ℝ) :
  geometric_sequence a →
  a 2 = 2 →
  a 3 * a 4 = 32 →
  a 8 = 128 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a8_l995_99597


namespace NUMINAMATH_CALUDE_even_sum_impossible_both_odd_l995_99594

theorem even_sum_impossible_both_odd (n m : ℤ) (h : Even (n^2 + m^2 + n*m)) : 
  ¬(Odd n ∧ Odd m) := by
  sorry

end NUMINAMATH_CALUDE_even_sum_impossible_both_odd_l995_99594


namespace NUMINAMATH_CALUDE_problem_1_l995_99586

theorem problem_1 (a b : ℝ) : a^2 + b^2 - 2*a + 1 = 0 → a = 1 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l995_99586


namespace NUMINAMATH_CALUDE_pyramid_volume_l995_99548

/-- Represents a pyramid with a square base ABCD and vertex E -/
structure Pyramid where
  base_area : ℝ
  triangle_ABE_area : ℝ
  triangle_CDE_area : ℝ

/-- The volume of the pyramid is 64√15 -/
theorem pyramid_volume (p : Pyramid) 
  (h1 : p.base_area = 64)
  (h2 : p.triangle_ABE_area = 48)
  (h3 : p.triangle_CDE_area = 40) :
  ∃ (v : ℝ), v = 64 * Real.sqrt 15 ∧ v = (1/3) * p.base_area * Real.sqrt ((2 * p.triangle_ABE_area / Real.sqrt p.base_area)^2 - (Real.sqrt p.base_area - 2 * p.triangle_CDE_area / Real.sqrt p.base_area)^2) := by
  sorry

end NUMINAMATH_CALUDE_pyramid_volume_l995_99548


namespace NUMINAMATH_CALUDE_frustum_volume_l995_99541

theorem frustum_volume (r₁ r₂ : ℝ) (h l : ℝ) : 
  r₂ = 4 * r₁ →
  l = 5 →
  π * (r₁ + r₂) * l = 25 * π →
  (1/3) * π * h * (r₁^2 + r₂^2 + r₁*r₂) = 28 * π :=
by sorry

end NUMINAMATH_CALUDE_frustum_volume_l995_99541


namespace NUMINAMATH_CALUDE_f_divisible_by_8_l995_99550

def f (n : ℕ) : ℤ := 5 * n + 2 * (-1)^n + 1

theorem f_divisible_by_8 : ∀ n : ℕ, ∃ k : ℤ, f n = 8 * k := by
  sorry

end NUMINAMATH_CALUDE_f_divisible_by_8_l995_99550


namespace NUMINAMATH_CALUDE_quadratic_always_real_roots_quadratic_distinct_positive_integer_roots_l995_99573

/-- The quadratic equation mx^2 - (m+2)x + 2 = 0 -/
def quadratic_equation (m : ℝ) (x : ℝ) : Prop :=
  m * x^2 - (m + 2) * x + 2 = 0

theorem quadratic_always_real_roots :
  ∀ m : ℝ, ∃ x : ℝ, quadratic_equation m x :=
sorry

theorem quadratic_distinct_positive_integer_roots :
  ∀ m : ℤ, (∃ x y : ℤ, x ≠ y ∧ 0 < x ∧ 0 < y ∧ quadratic_equation (m : ℝ) (x : ℝ) ∧ quadratic_equation (m : ℝ) (y : ℝ)) ↔ m = 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_always_real_roots_quadratic_distinct_positive_integer_roots_l995_99573


namespace NUMINAMATH_CALUDE_input_is_input_statement_l995_99578

-- Define the type for programming language statements
inductive Statement
  | Print
  | Input
  | If
  | Let

-- Define a predicate for input statements
def isInputStatement : Statement → Prop
  | Statement.Input => True
  | _ => False

-- Theorem: INPUT is an input statement
theorem input_is_input_statement : isInputStatement Statement.Input := by
  sorry

end NUMINAMATH_CALUDE_input_is_input_statement_l995_99578


namespace NUMINAMATH_CALUDE_total_apples_l995_99516

/-- Represents the number of apples in a pack -/
def apples_per_pack : ℕ := 4

/-- Represents the number of packs bought -/
def packs_bought : ℕ := 2

/-- Theorem stating that buying 2 packs of 4 apples each results in 8 apples total -/
theorem total_apples : apples_per_pack * packs_bought = 8 := by
  sorry

end NUMINAMATH_CALUDE_total_apples_l995_99516


namespace NUMINAMATH_CALUDE_max_wellfed_pikes_l995_99558

/-- Represents the state of pikes in a pond -/
structure PikeState where
  total : ℕ
  wellfed : ℕ
  hungry : ℕ

/-- Defines what it means for a pike state to be valid -/
def is_valid_state (s : PikeState) : Prop :=
  s.total = s.wellfed + s.hungry ∧ s.wellfed * 3 + s.hungry ≤ 40

/-- Defines what it means for a pike state to be maximal -/
def is_maximal_state (s : PikeState) : Prop :=
  is_valid_state s ∧ ∀ t : PikeState, is_valid_state t → t.wellfed ≤ s.wellfed

/-- The theorem to be proved -/
theorem max_wellfed_pikes :
  ∃ s : PikeState, is_maximal_state s ∧ s.wellfed = 13 := by
  sorry

end NUMINAMATH_CALUDE_max_wellfed_pikes_l995_99558


namespace NUMINAMATH_CALUDE_haploid_breeding_shortens_cycle_l995_99519

/-- Represents a haploid organism -/
structure Haploid where
  chromosomeSets : ℕ
  derivedFromGamete : Bool

/-- Represents the process of haploid breeding -/
structure HaploidBreeding where
  usesPollenGrains : Bool
  inducesChromosomeDoubling : Bool

/-- Represents the outcome of breeding -/
structure BreedingOutcome where
  cycleLength : ℕ
  homozygosity : Bool

/-- Theorem stating that haploid breeding shortens the breeding cycle -/
theorem haploid_breeding_shortens_cycle 
  (h : Haploid) 
  (hb : HaploidBreeding) 
  (outcome : BreedingOutcome) : 
  h.derivedFromGamete ∧ 
  hb.usesPollenGrains ∧ 
  hb.inducesChromosomeDoubling ∧
  outcome.homozygosity → 
  outcome.cycleLength < regular_breeding_cycle_length :=
sorry

/-- The length of a regular breeding cycle -/
def regular_breeding_cycle_length : ℕ := sorry

end NUMINAMATH_CALUDE_haploid_breeding_shortens_cycle_l995_99519


namespace NUMINAMATH_CALUDE_base_number_proof_l995_99530

theorem base_number_proof (x : ℝ) : x^3 = 1024 * (1/4)^2 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_base_number_proof_l995_99530


namespace NUMINAMATH_CALUDE_stratified_sampling_female_count_l995_99556

theorem stratified_sampling_female_count :
  ∀ (total_male total_female : ℕ) (male_prob : ℚ),
    total_male = 28 →
    total_female = 21 →
    male_prob = 1/7 →
    (total_female : ℚ) * male_prob = 3 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_female_count_l995_99556


namespace NUMINAMATH_CALUDE_exists_unassemblable_configuration_l995_99546

/-- Represents a rhombus divided into two triangles --/
structure Rhombus :=
  (white : Bool) -- true if the left triangle is white, false if it's gray

/-- Represents a rotation of the rhombus --/
inductive Rotation
  | R0   -- No rotation
  | R90  -- 90 degrees clockwise
  | R180 -- 180 degrees
  | R270 -- 270 degrees clockwise

/-- Represents a position in a 2D grid --/
structure Position :=
  (x : Int) (y : Int)

/-- Represents a placed rhombus in a configuration --/
structure PlacedRhombus :=
  (rhombus : Rhombus)
  (rotation : Rotation)
  (position : Position)

/-- Represents a configuration of rhombuses --/
def Configuration := List PlacedRhombus

/-- Checks if a configuration is valid --/
def isValidConfiguration (config : Configuration) : Bool :=
  sorry

/-- Checks if a larger shape can be assembled from the given rhombuses --/
def canAssembleLargerShape (shape : Configuration) : Bool :=
  sorry

/-- The main theorem stating that there exists a configuration that cannot be assembled --/
theorem exists_unassemblable_configuration :
  ∃ (shape : Configuration),
    isValidConfiguration shape ∧ ¬canAssembleLargerShape shape :=
  sorry

end NUMINAMATH_CALUDE_exists_unassemblable_configuration_l995_99546


namespace NUMINAMATH_CALUDE_x_equation_l995_99514

theorem x_equation (x : ℝ) (h : x + 1/x = Real.sqrt 5) : x^11 - 7*x^7 + x^3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_x_equation_l995_99514


namespace NUMINAMATH_CALUDE_range_of_x_l995_99543

theorem range_of_x (x : ℝ) : 
  ¬((x ∈ Set.Icc 2 5) ∨ (x < 1 ∨ x > 4)) → x ∈ Set.Ico 1 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_x_l995_99543


namespace NUMINAMATH_CALUDE_class_gender_ratio_l995_99547

/-- The ratio of boys to girls in a class based on probabilities of correct answers -/
theorem class_gender_ratio 
  (α : ℝ) -- probability of teacher's correct answer
  (β : ℝ) -- probability of boy's correct answer
  (γ : ℝ) -- probability of girl's correct answer
  (h_prob : ∀ (x y : ℝ), x / (x + y) * β + y / (x + y) * γ = 1/2) -- probability condition
  : (α ≠ 1/2 → ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x / y = (1/2 - γ) / (β - 1/2)) ∧ 
    ((α = 1/2 ∨ (β = 1/2 ∧ γ = 1/2)) → ∀ (r : ℝ), r > 0 → ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x / y = r) :=
by sorry

end NUMINAMATH_CALUDE_class_gender_ratio_l995_99547


namespace NUMINAMATH_CALUDE_smallest_n_proof_l995_99520

/-- The smallest possible value of n given the conditions -/
def smallest_n : ℕ := 400

theorem smallest_n_proof (a b c m n : ℕ) 
  (h1 : a > 0 ∧ b > 0 ∧ c > 0)
  (h2 : a + b + c = 2003)
  (h3 : a = 2 * b)
  (h4 : a.factorial * b.factorial * c.factorial = m * (10 ^ n))
  (h5 : ¬(10 ∣ m)) : 
  n ≥ smallest_n := by
  sorry

#check smallest_n_proof

end NUMINAMATH_CALUDE_smallest_n_proof_l995_99520


namespace NUMINAMATH_CALUDE_min_value_sum_l995_99536

theorem min_value_sum (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h_pos : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0 ∧ x₅ > 0)
  (h_sum : x₁^3 + x₂^3 + x₃^3 + x₄^3 + x₅^3 = 1) : 
  x₁/(1 - x₁^2) + x₂/(1 - x₂^2) + x₃/(1 - x₃^2) + x₄/(1 - x₄^2) + x₅/(1 - x₅^2) ≥ 3 * Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_l995_99536


namespace NUMINAMATH_CALUDE_lisa_caffeine_over_goal_l995_99529

/-- The amount of caffeine Lisa consumed over her goal -/
def caffeine_over_goal (caffeine_per_cup : ℕ) (daily_limit : ℕ) (cups_drunk : ℕ) : ℕ :=
  (caffeine_per_cup * cups_drunk) - daily_limit

theorem lisa_caffeine_over_goal :
  let caffeine_per_cup : ℕ := 80
  let daily_limit : ℕ := 200
  let cups_drunk : ℕ := 3
  caffeine_over_goal caffeine_per_cup daily_limit cups_drunk = 40 := by
sorry

end NUMINAMATH_CALUDE_lisa_caffeine_over_goal_l995_99529


namespace NUMINAMATH_CALUDE_addition_and_subtraction_proof_l995_99500

theorem addition_and_subtraction_proof :
  (1 + (-11) = -10) ∧ (0 - 4.5 = -4.5) := by sorry

end NUMINAMATH_CALUDE_addition_and_subtraction_proof_l995_99500


namespace NUMINAMATH_CALUDE_fraction_sum_theorem_l995_99521

theorem fraction_sum_theorem (a b c : ℝ) 
  (h1 : a * c / (a + b) + b * a / (b + c) + c * b / (c + a) = 3)
  (h2 : b * c / (a + b) + c * a / (b + c) + a * b / (c + a) = -4) :
  b / (a + b) + c / (b + c) + a / (c + a) = 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_theorem_l995_99521


namespace NUMINAMATH_CALUDE_range_of_a_l995_99595

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then -x^2 + 3*x else Real.log (x + 1)

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x, |f x| ≥ a * x) ↔ a ∈ Set.Icc (-3 : ℝ) 0 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l995_99595


namespace NUMINAMATH_CALUDE_greatest_multiple_of_four_l995_99572

theorem greatest_multiple_of_four (x : ℕ) : 
  x > 0 → 
  ∃ k : ℕ, x = 4 * k → 
  x^2 < 500 → 
  ∀ y : ℕ, (y > 0 ∧ ∃ m : ℕ, y = 4 * m ∧ y^2 < 500) → y ≤ 20 :=
by sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_four_l995_99572


namespace NUMINAMATH_CALUDE_infinite_primes_dividing_polynomial_values_l995_99503

/-- A polynomial with integer coefficients -/
def IntPolynomial := List ℤ

/-- The degree of a polynomial -/
def degree (p : IntPolynomial) : ℕ :=
  (p.length - 1).max 0

/-- Evaluate a polynomial at a given integer -/
def eval (p : IntPolynomial) (x : ℤ) : ℤ :=
  p.enum.foldl (fun acc (i, a) => acc + a * x ^ i) 0

theorem infinite_primes_dividing_polynomial_values (p : IntPolynomial)
  (h : degree p ≥ 1) :
  ∃ (S : Set ℕ), Set.Infinite S ∧
    (∀ q ∈ S, Prime q ∧ ∃ n : ℕ, (eval p n) % q = 0) :=
  sorry

end NUMINAMATH_CALUDE_infinite_primes_dividing_polynomial_values_l995_99503


namespace NUMINAMATH_CALUDE_ellipse_left_vertex_l995_99509

-- Define the ellipse
def is_ellipse (a b x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the circle
def is_circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x + 8 = 0

-- Theorem statement
theorem ellipse_left_vertex 
  (a b : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : ∃ (x y : ℝ), is_circle x y ∧ is_ellipse a b (x - 3) y) 
  (h4 : 2 * b = 8) :
  is_ellipse a b (-5) 0 := by sorry

end NUMINAMATH_CALUDE_ellipse_left_vertex_l995_99509


namespace NUMINAMATH_CALUDE_no_valid_A_l995_99531

theorem no_valid_A : ¬∃ A : ℕ, 
  0 ≤ A ∧ A ≤ 9 ∧ 
  45 % A = 0 ∧ 
  (3571 * 10 + A) * 10 + 6 % 4 = 0 ∧ 
  (3571 * 10 + A) * 10 + 6 % 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_valid_A_l995_99531


namespace NUMINAMATH_CALUDE_hyperbola_equation_l995_99574

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if its eccentricity is 5/3 and the distance from its right focus to one asymptote is 4,
    then a = 3 and b = 4 -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  let e := 5/3  -- eccentricity
  let d := 4    -- distance from right focus to asymptote
  let c := Real.sqrt (a^2 + b^2)  -- distance from center to focus
  e = c/a ∧ d = b → a = 3 ∧ b = 4 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l995_99574


namespace NUMINAMATH_CALUDE_conjugate_complex_modulus_l995_99504

theorem conjugate_complex_modulus (α β : ℂ) :
  (∃ (x y : ℝ), α = x + y * I ∧ β = x - y * I) →  -- α and β are conjugates
  (α^2 / β).im = 0 →                              -- α²/β is real
  Complex.abs (α - β) = 4 →                       -- |α - β| = 4
  Complex.abs α = 4 * Real.sqrt 3 / 3 :=          -- |α| = 4√3/3
by sorry

end NUMINAMATH_CALUDE_conjugate_complex_modulus_l995_99504


namespace NUMINAMATH_CALUDE_quadratic_inequality_problem_l995_99508

/-- Given that the solution set of ax^2 + 5x - 2 > 0 is {x | 1/2 < x < 2}, 
    prove the value of a and the solution set of ax^2 - 5x + a^2 - 1 > 0 -/
theorem quadratic_inequality_problem 
  (h : ∀ x : ℝ, ax^2 + 5*x - 2 > 0 ↔ 1/2 < x ∧ x < 2) :
  (a = -2) ∧ 
  (∀ x : ℝ, a*x^2 - 5*x + a^2 - 1 > 0 ↔ -3 < x ∧ x < 1/2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_problem_l995_99508


namespace NUMINAMATH_CALUDE_repeating_decimal_calculation_l995_99561

/-- Represents a repeating decimal where the digits after the decimal point repeat indefinitely -/
def repeating_decimal (n : ℕ) (d : ℕ) : ℚ := n / d

theorem repeating_decimal_calculation :
  let x := repeating_decimal 27 100000
  (10^5 - 10^3) * x = 26.73 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_calculation_l995_99561


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l995_99549

/-- Given a rectangle with area 3a² - 3ab + 6a and one side length 3a, its perimeter is 8a - 2b + 4 -/
theorem rectangle_perimeter (a b : ℝ) : 
  let area := 3 * a^2 - 3 * a * b + 6 * a
  let side1 := 3 * a
  let side2 := area / side1
  let perimeter := 2 * (side1 + side2)
  perimeter = 8 * a - 2 * b + 4 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l995_99549


namespace NUMINAMATH_CALUDE_birthday_puzzle_l995_99518

theorem birthday_puzzle :
  ∃! (X Y : ℕ), 
    31 * X + 12 * Y = 376 ∧
    1 ≤ X ∧ X ≤ 12 ∧
    1 ≤ Y ∧ Y ≤ 31 ∧
    X = 9 ∧ Y = 8 := by
  sorry

end NUMINAMATH_CALUDE_birthday_puzzle_l995_99518


namespace NUMINAMATH_CALUDE_triangle_abc_area_l995_99579

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that its area is (√3 + 1) / 2 under the given conditions. -/
theorem triangle_abc_area (A B C : Real) (a b c : Real) :
  b = Real.sqrt 2 * a →
  Real.sqrt 3 * Real.cos B = Real.sqrt 2 * Real.cos A →
  c = Real.sqrt 3 + 1 →
  (1/2) * a * c * Real.sin B = (Real.sqrt 3 + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_area_l995_99579


namespace NUMINAMATH_CALUDE_bracket_two_equals_twelve_l995_99583

def bracket (x : ℝ) : ℝ := x^2 + 2*x + 4

theorem bracket_two_equals_twelve : bracket 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_bracket_two_equals_twelve_l995_99583


namespace NUMINAMATH_CALUDE_pet_store_birds_pet_store_birds_solution_l995_99542

/-- Proves that the initial number of birds in the pet store was 12 -/
theorem pet_store_birds : ℕ → Prop :=
  fun initial_birds =>
    let initial_puppies : ℕ := 9
    let initial_cats : ℕ := 5
    let initial_spiders : ℕ := 15
    let remaining_birds : ℕ := initial_birds / 2
    let remaining_puppies : ℕ := initial_puppies - 3
    let remaining_cats : ℕ := initial_cats
    let remaining_spiders : ℕ := initial_spiders - 7
    let total_remaining : ℕ := remaining_birds + remaining_puppies + remaining_cats + remaining_spiders
    total_remaining = 25 → initial_birds = 12

/-- The theorem holds for 12 initial birds -/
theorem pet_store_birds_solution : pet_store_birds 12 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_birds_pet_store_birds_solution_l995_99542


namespace NUMINAMATH_CALUDE_dimes_borrowed_proof_l995_99527

/-- Represents the number of dimes Fred had initially -/
def initial_dimes : ℕ := 7

/-- Represents the number of dimes Fred has left -/
def remaining_dimes : ℕ := 4

/-- Represents the number of dimes Fred's sister borrowed -/
def borrowed_dimes : ℕ := initial_dimes - remaining_dimes

/-- Proves that the number of borrowed dimes is equal to the difference between
    the initial number of dimes and the remaining number of dimes -/
theorem dimes_borrowed_proof :
  borrowed_dimes = initial_dimes - remaining_dimes := by
  sorry

end NUMINAMATH_CALUDE_dimes_borrowed_proof_l995_99527


namespace NUMINAMATH_CALUDE_negative_expressions_l995_99517

/-- Represents a real number with an approximate value -/
structure ApproxReal where
  value : ℝ
  approx : ℝ
  is_close : |value - approx| < 0.1

/-- Given approximate values for U, V, W, X, and Y, prove which expressions are negative -/
theorem negative_expressions 
  (U : ApproxReal) (hU : U.approx = -4.6)
  (V : ApproxReal) (hV : V.approx = -2.0)
  (W : ApproxReal) (hW : W.approx = 0.2)
  (X : ApproxReal) (hX : X.approx = 1.3)
  (Y : ApproxReal) (hY : Y.approx = 2.2) :
  U.value - V.value < 0 ∧ 
  (X.value / V.value) * U.value < 0 ∧ 
  U.value * V.value ≥ 0 ∧ 
  W.value / (U.value * V.value) ≥ 0 ∧ 
  (X.value + Y.value) / W.value ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_negative_expressions_l995_99517
