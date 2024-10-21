import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_k_cubed_divisible_by_336_l241_24133

theorem least_k_cubed_divisible_by_336 : ∃ k : ℕ, 
  (∀ m : ℕ, m^3 % 336 = 0 → k ≤ m) ∧ k^3 % 336 = 0 ∧ k = 84 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_k_cubed_divisible_by_336_l241_24133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_card_set_satisfies_conditions_l241_24163

def card_set : Finset ℚ := {1/2, 3/2, 5/2, 5/2, 7/2, 9/2, 9/2, 11/2, 13/2}

def dice_sum_freq : Fin 11 → ℕ
| ⟨0, _⟩ => 1  -- sum of 2
| ⟨1, _⟩ => 2  -- sum of 3
| ⟨2, _⟩ => 3  -- sum of 4
| ⟨3, _⟩ => 4  -- sum of 5
| ⟨4, _⟩ => 5  -- sum of 6
| ⟨5, _⟩ => 6  -- sum of 7
| ⟨6, _⟩ => 5  -- sum of 8
| ⟨7, _⟩ => 4  -- sum of 9
| ⟨8, _⟩ => 3  -- sum of 10
| ⟨9, _⟩ => 2  -- sum of 11
| ⟨10, _⟩ => 1 -- sum of 12

theorem card_set_satisfies_conditions :
  (card_set.card = 9) ∧
  (∀ x ∈ card_set, ∃ (n : ℕ), x = n / 2) ∧
  (∀ i : Fin 11, (card_set.filter (λ x => ∃ y ∈ card_set, x + y = (i.val + 2 : ℚ))).card = dice_sum_freq i) :=
by sorry

#eval card_set
#eval card_set.card

end NUMINAMATH_CALUDE_ERRORFEEDBACK_card_set_satisfies_conditions_l241_24163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_union_possible_values_of_a_l241_24149

-- Define the sets A and B
def A : Set ℝ := {x | 2 * x^2 - x - 6 ≥ 0}
def B : Set ℝ := {x | x > 0 ∧ Real.log x / Real.log 2 ≤ 2}

-- Define the set C
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

-- Theorem for part (1)
theorem intersection_and_union :
  (A ∩ B = {x | 2 ≤ x ∧ x ≤ 4}) ∧
  ((Set.univ \ B) ∪ A = {x | x ≤ 0 ∨ x ≥ 2}) := by sorry

-- Theorem for part (2)
theorem possible_values_of_a :
  {a : ℝ | ∀ x, x ∈ C a → x ∈ B} = {a | 0 ≤ a ∧ a ≤ 3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_union_possible_values_of_a_l241_24149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_is_odd_but_f_is_not_l241_24131

-- Define the sine function
noncomputable def sine : ℝ → ℝ := Real.sin

-- Define the property of being an odd function
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define our specific function f(x) = sin(x^2 + 2)
noncomputable def f (x : ℝ) : ℝ := sine (x^2 + 2)

-- State the theorem
theorem sine_is_odd_but_f_is_not :
  (is_odd sine) ∧ ¬(is_odd f) := by
  sorry

#check sine_is_odd_but_f_is_not

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_is_odd_but_f_is_not_l241_24131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_to_cuboid_volume_ratio_l241_24110

/-- Conversion factor from centimeters to meters -/
noncomputable def cm_to_m : ℝ := 1 / 100

/-- Volume of a cube with edge length 1 meter -/
def cube_volume : ℝ := 1

/-- Width of the cuboid in centimeters -/
def cuboid_width_cm : ℝ := 50

/-- Length of the cuboid in centimeters -/
def cuboid_length_cm : ℝ := 50

/-- Height of the cuboid in centimeters -/
def cuboid_height_cm : ℝ := 20

/-- Volume of the cuboid in cubic meters -/
noncomputable def cuboid_volume : ℝ :=
  (cuboid_width_cm * cm_to_m) * (cuboid_length_cm * cm_to_m) * (cuboid_height_cm * cm_to_m)

/-- Theorem stating the ratio of cube volume to cuboid volume -/
theorem cube_to_cuboid_volume_ratio :
  cube_volume / cuboid_volume = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_to_cuboid_volume_ratio_l241_24110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diver_descent_rate_l241_24132

/-- Given a diver's descent, calculate the rate of descent. -/
noncomputable def rate_of_descent (depth : ℝ) (time : ℝ) : ℝ :=
  depth / time

/-- The depth of the dive in feet. -/
def total_depth : ℝ := 6400

/-- The time taken for the dive in minutes. -/
def total_time : ℝ := 200

/-- Theorem stating that the diver's descent rate is 32 feet per minute. -/
theorem diver_descent_rate :
  rate_of_descent total_depth total_time = 32 := by
  -- Unfold the definitions
  unfold rate_of_descent total_depth total_time
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diver_descent_rate_l241_24132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_reach_window_proof_l241_24145

/-- Time to reach the ticket window -/
noncomputable def time_to_reach_window (initial_distance : ℝ) (time_spent : ℝ) (distance_moved : ℝ) : ℝ :=
  let yards_to_feet := 3
  let initial_distance_feet := initial_distance * yards_to_feet
  let remaining_distance := initial_distance_feet - distance_moved
  let rate := distance_moved / time_spent
  remaining_distance / rate

/-- Theorem: The time to reach the ticket window is 280/3 minutes -/
theorem time_to_reach_window_proof (initial_distance : ℝ) (time_spent : ℝ) (distance_moved : ℝ)
  (h1 : initial_distance = 100)
  (h2 : time_spent = 40)
  (h3 : distance_moved = 90) :
  time_to_reach_window initial_distance time_spent distance_moved = 280/3 := by
  sorry

#eval (280 : ℚ) / 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_reach_window_proof_l241_24145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lock_can_be_opened_l241_24164

/-- Represents a 4x4 array of binary states -/
def LockState := Fin 4 → Fin 4 → Bool

/-- Represents the operation of switching a key at a given position -/
def switch (i j : Fin 4) (state : LockState) : LockState :=
  fun x y => if x = i ∨ y = j then !(state x y) else state x y

/-- Checks if all keys in the lock state are vertical (represented by false) -/
def allVertical (state : LockState) : Prop :=
  ∀ i j, state i j = false

/-- States that any lock configuration can be opened -/
theorem lock_can_be_opened (initial : LockState) : 
  ∃ (sequence : List (Fin 4 × Fin 4)), allVertical (sequence.foldl (fun st (ij) => switch ij.1 ij.2 st) initial) := by
  sorry

#check lock_can_be_opened

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lock_can_be_opened_l241_24164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_angles_less_than_1100_l241_24194

/-- Represents the scenario of a car passing a fence -/
structure CarPassingFence where
  carSpeed : ℝ  -- Speed of the car in km/h
  fenceLength : ℝ  -- Length of the fence in meters
  measurementInterval : ℝ  -- Interval between measurements in seconds

/-- Function to calculate the sum of measured angles (implementation not provided) -/
noncomputable def sum_of_measured_angles (scenario : CarPassingFence) : ℝ :=
  sorry

/-- Theorem stating that the sum of angles measured is less than 1100 degrees -/
theorem sum_of_angles_less_than_1100 (scenario : CarPassingFence)
  (h1 : scenario.carSpeed = 60)
  (h2 : scenario.fenceLength = 100)
  (h3 : scenario.measurementInterval = 1) :
  ∃ (sumOfAngles : ℝ), sumOfAngles < 1100 ∧ 
  sumOfAngles = sum_of_measured_angles scenario :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_angles_less_than_1100_l241_24194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_construction_agency_laborers_l241_24142

theorem construction_agency_laborers
  (total_hired : ℕ) (total_payroll : ℕ) (operator_pay : ℕ) (laborer_pay : ℕ) (laborers : ℕ) :
  total_hired = 35 →
  total_payroll = 3950 →
  operator_pay = 140 →
  laborer_pay = 90 →
  let operators := total_hired - laborers
  (operators + laborers = total_hired) →
  (operator_pay * operators + laborer_pay * laborers = total_payroll) →
  laborers = 19 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_construction_agency_laborers_l241_24142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tillys_bag_cost_l241_24184

/-- Calculates the cost per bag for Tilly's business scenario -/
theorem tillys_bag_cost (num_bags : ℕ) (selling_price profit shipping_fee : ℚ) 
  (sales_tax_rate : ℚ) : 
  num_bags = 100 →
  selling_price = 10 →
  profit = 300 →
  sales_tax_rate = 5 / 100 →
  shipping_fee = 50 →
  (↑num_bags * selling_price - (profit + (↑num_bags * (selling_price * sales_tax_rate) + shipping_fee))) / ↑num_bags = 6 := by
  sorry

#check tillys_bag_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tillys_bag_cost_l241_24184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_exists_in_interval_l241_24111

-- Define the function f(x) = ln(x+1) - 2/x
noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1) - 2 / x

-- State the theorem
theorem root_exists_in_interval :
  ∃ c ∈ Set.Ioo 1 2, f c = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_exists_in_interval_l241_24111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quiz_challenge_theorem_l241_24196

/-- Represents the score of a team in a quiz challenge -/
inductive Score : Type
  | MinusOne : Score
  | Zero : Score
  | One : Score

/-- Represents the probability distribution of the team's score -/
noncomputable def score_distribution (p_a p_b : ℝ) : Score → ℝ
  | Score.MinusOne => (1 - p_a) * (1 - p_b)
  | Score.Zero => p_a * (1 - p_b) + (1 - p_a) * p_b
  | Score.One => p_a * p_b

/-- Calculates the expected value of the team's score -/
noncomputable def expected_score (p_a p_b : ℝ) : ℝ :=
  -1 * score_distribution p_a p_b Score.MinusOne +
   0 * score_distribution p_a p_b Score.Zero +
   1 * score_distribution p_a p_b Score.One

/-- Represents the probability of not getting 1 point in three consecutive rounds -/
noncomputable def P : ℕ → ℝ → ℝ → ℝ → ℝ
  | 0, _, _, _ => 1
  | 1, _, _, _ => 1
  | 2, _, _, _ => 1
  | 3, _, _, _ => 1 - (1/2)^3
  | n + 4, a, b, c => a * P (n + 3) a b c + b * P (n + 2) a b c + c * P (n + 1) a b c

theorem quiz_challenge_theorem (p_a p_b a b c : ℝ)
  (h_pa : p_a = 3/4)
  (h_pb : p_b = 2/3)
  (h_a : a = 1/2)
  (h_b : b = 1/4)
  (h_c : c = 1/8) :
  expected_score p_a p_b = 5/12 ∧
  (∀ n : ℕ, n ≥ 3 → P (n + 1) a b c < P n a b c) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quiz_challenge_theorem_l241_24196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_height_difference_packing_methods_l241_24170

/-- The diameter of each cylindrical pipe in centimeters -/
def pipe_diameter : ℝ := 12

/-- The number of pipes in each crate -/
def num_pipes : ℕ := 180

/-- The height of the crate with simple vertical stacking in centimeters -/
def height_simple_stacking : ℝ := 216

/-- The height of the crate with staggered (hexagonal) packing in centimeters -/
noncomputable def height_staggered_packing : ℝ := 12 + 102 * Real.sqrt 3

/-- The theorem stating the positive difference in heights between the two packing methods -/
theorem height_difference_packing_methods :
  |height_simple_stacking - height_staggered_packing| = 204 - 102 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_height_difference_packing_methods_l241_24170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_transformation_l241_24193

theorem sin_cos_transformation :
  ∃ k : ℝ, 
    (∀ x : ℝ, Real.sin (3 * x) - Real.sqrt 3 * Real.cos (3 * x) = 2 * Real.sin (3 * x - π / 3)) ∧ 
    (∀ x : ℝ, 2 * Real.sin (3 * x - π / 3) = 2 * Real.cos (3 * (x - k))) ∧ 
    k = 5 * π / 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_transformation_l241_24193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_heptagon_with_congruent_diagonals_is_regular_l241_24134

-- Define a 7-gon
structure Heptagon where
  vertices : Fin 7 → ℝ × ℝ

-- Define a function to represent diagonals
def diagonal (H : Heptagon) (i j : Fin 7) : ℝ × ℝ :=
  (H.vertices j.1 - H.vertices i.1)

-- Define congruence for diagonals
def congruent_diagonals (H : Heptagon) (diags : List (Fin 7 × Fin 7)) : Prop :=
  ∀ (i j k l : Fin 7), (i, j) ∈ diags → (k, l) ∈ diags →
    (diagonal H i j).1^2 + (diagonal H i j).2^2 = (diagonal H k l).1^2 + (diagonal H k l).2^2

-- Define regularity for a heptagon
def is_regular (H : Heptagon) : Prop :=
  ∀ (i j : Fin 7), i ≠ j →
    (H.vertices (i + 1)).1^2 + (H.vertices (i + 1)).2^2 =
    (H.vertices (j + 1)).1^2 + (H.vertices (j + 1)).2^2

-- State the theorem
theorem heptagon_with_congruent_diagonals_is_regular (H : Heptagon) :
  congruent_diagonals H [(1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 1), (7, 2)] ∧
  congruent_diagonals H [(1, 4), (2, 5), (3, 6), (4, 7), (5, 1), (6, 2), (7, 3)] →
  is_regular H := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_heptagon_with_congruent_diagonals_is_regular_l241_24134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_line_theorem_l241_24104

-- Define the points
variable (A B C D E F : ℝ × ℝ)

-- Define the parallelogram property
def is_parallelogram (A B C D : ℝ × ℝ) : Prop :=
  (B.1 - A.1, B.2 - A.2) = (D.1 - C.1, D.2 - C.2) ∧
  (C.1 - B.1, C.2 - B.2) = (A.1 - D.1, A.2 - D.2)

-- Define the collinearity property
def collinear (P Q R : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, R.1 - P.1 = t * (Q.1 - P.1) ∧ R.2 - P.2 = t * (Q.2 - P.2)

-- Define the distance between two points
noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)

-- State the theorem
theorem parallelogram_line_theorem
  (h_parallelogram : is_parallelogram A B C D)
  (h_collinear_E : collinear A B E)
  (h_collinear_F : collinear A D F)
  (h_C_outside : ¬collinear C A B ∧ ¬collinear C A D) :
  (distance A C)^2 + (distance C E) * (distance C F) =
  (distance A B) * (distance A E) + (distance A D) * (distance A F) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_line_theorem_l241_24104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_n_squared_l241_24105

def has_exactly_four_divisors (n : ℕ) : Prop :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card = 4

theorem divisors_of_n_squared (n : ℕ) (h : has_exactly_four_divisors n) :
  let d := (Finset.filter (· ∣ n^2) (Finset.range (n^2 + 1))).card
  d = 7 ∨ d = 9 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_n_squared_l241_24105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angles_form_triangle_l241_24123

/-- Given angles α, β, γ satisfying specific cosine relations, prove they form a triangle --/
theorem angles_form_triangle (α β γ : Real) 
  (h₁ : Real.cos α = Real.sqrt 2 / (4 * Real.cos (10 * Real.pi / 180)))
  (h₂ : Real.cos β = Real.sqrt 6 / (4 * Real.cos (10 * Real.pi / 180)))
  (h₃ : Real.cos γ = 1 / (2 * Real.cos (10 * Real.pi / 180))) :
  α + β + γ = Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angles_form_triangle_l241_24123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_even_probability_l241_24144

def box := Finset.range 4

theorem product_even_probability :
  let outcomes := box.product box
  let favorable := outcomes.filter (fun p => Even ((p.1 + 1) * (p.2 + 1)))
  (favorable.card : ℚ) / outcomes.card = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_even_probability_l241_24144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_gcd_2015_l241_24146

theorem max_gcd_2015 (x y : ℤ) (h : Int.gcd x y = 1) :
  (∃ a b : ℤ, Int.gcd (a + 2015 * b) (b + 2015 * a) = 4060224) ∧
  (∀ c d : ℤ, Int.gcd (c + 2015 * d) (d + 2015 * c) ≤ 4060224) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_gcd_2015_l241_24146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_even_l241_24198

noncomputable def g (x : ℝ) : ℝ := ((3^x - 1) / (3^x + 1))^2

theorem g_is_even : ∀ x : ℝ, g (-x) = g x := by
  intro x
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_even_l241_24198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sarah_cake_count_l241_24180

def is_valid_cake_count (c : ℕ) : Prop :=
  ∃ k : ℕ,
    c + k = 5 ∧
    (90 * c + 40 * k) % 100 = 0

theorem sarah_cake_count :
  ∃ c : ℕ, c ∈ ({0, 2, 4} : Set ℕ) ∧ is_valid_cake_count c ∧
  ∀ x : ℕ, x ∉ ({0, 2, 4} : Set ℕ) → ¬is_valid_cake_count x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sarah_cake_count_l241_24180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sea_glass_ratio_l241_24155

/-- Sea glass collection problem -/
theorem sea_glass_ratio :
  ∃ (blanche_green blanche_red rose_red rose_blue dorothy_total : ℕ),
  let dorothy_red := 2 * (blanche_red + rose_red);
  let dorothy_blue := dorothy_total - dorothy_red;
  let ratio : ℚ := dorothy_blue / rose_blue;
  blanche_green = 12 ∧
  blanche_red = 3 ∧
  rose_red = 9 ∧
  rose_blue = 11 ∧
  dorothy_total = 57 ∧
  ratio = 3 := by
  -- Prove the existence of the numbers
  use 12, 3, 9, 11, 57
  -- Simplify the goal
  simp [Nat.cast_add, Nat.cast_mul, Nat.cast_sub]
  -- Perform the calculation
  norm_num
  -- QED
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sea_glass_ratio_l241_24155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_card_access_theorem_l241_24187

/-- Represents a credit card with a PIN and attempts left -/
structure CreditCard where
  pin : Nat
  attemptsLeft : Nat

/-- Represents the state of Kirpich's attempt to access the cards -/
structure CardAccessState where
  cards : List CreditCard
  pins : List Nat
  successfulAccesses : Nat

/-- The initial state with 4 cards, 4 PINs, and 3 attempts per card -/
def initialState : CardAccessState :=
  { cards := List.replicate 4 { pin := 0, attemptsLeft := 3 },
    pins := [0, 1, 2, 3],
    successfulAccesses := 0 }

/-- Predicate to check if at least 3 cards can be accessed -/
def canAccessAtLeastThree (state : CardAccessState) : Prop :=
  state.successfulAccesses ≥ 3

/-- The probability of accessing all 4 cards successfully -/
def probabilityAccessAll : ℚ := 1 / 24

theorem card_access_theorem (state : CardAccessState) :
  state = initialState →
  (∃ finalState, canAccessAtLeastThree finalState) ∧
  probabilityAccessAll = 1 / 24 := by
  sorry

#check card_access_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_card_access_theorem_l241_24187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_center_distance_l241_24148

/-- The distance traveled by the center of a cylinder rolling along a path of three quarter-circle arcs -/
theorem cylinder_center_distance (cylinder_diameter : ℝ) (arc1_radius arc2_radius arc3_radius : ℝ) : 
  cylinder_diameter = 6 →
  arc1_radius = 150 →
  arc2_radius = 90 →
  arc3_radius = 120 →
  (π / 2) * ((arc1_radius + cylinder_diameter / 2) + 
             (arc2_radius - cylinder_diameter / 2) + 
             (arc3_radius + cylinder_diameter / 2)) = 181.5 * π := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_center_distance_l241_24148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stepmother_winning_condition_l241_24106

/-- Represents the game state with five buckets arranged in a pentagon -/
structure GameState where
  buckets : Fin 5 → ℝ
  capacity : ℝ

/-- The Stepmother's strategy for distributing water -/
def StepmotherStrategy := GameState → Fin 5 → ℝ

/-- Cinderella's strategy for choosing which adjacent buckets to empty -/
def CinderellaStrategy := GameState → Fin 5

/-- Determines if a given game state has any overflowing bucket -/
def hasOverflow (state : GameState) : Prop :=
  ∃ i, state.buckets i > state.capacity

/-- Simulates one round of the game -/
def playRound (state : GameState) (stepStrat : StepmotherStrategy) (cinderStrat : CinderellaStrategy) : GameState :=
  sorry

/-- Determines if the Stepmother has a winning strategy for a given capacity -/
def stepmotherWins (capacity : ℝ) : Prop :=
  ∃ (stepStrat : StepmotherStrategy), ∀ (cinderStrat : CinderellaStrategy),
    ∃ (n : ℕ), hasOverflow (Nat.iterate (playRound · stepStrat cinderStrat) n { buckets := λ _ => 0, capacity })

/-- The main theorem stating the condition for the Stepmother's win -/
theorem stepmother_winning_condition (capacity : ℝ) :
  stepmotherWins capacity ↔ capacity < 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stepmother_winning_condition_l241_24106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_property_l241_24171

-- Define the function f(x) = x sin(x)
noncomputable def f (x : ℝ) : ℝ := x * Real.sin x

-- Define the derivative of f(x)
noncomputable def f_derivative (x : ℝ) : ℝ := Real.sin x + x * Real.cos x

-- Theorem statement
theorem extreme_value_property (x₀ : ℝ) :
  (∃ (ε : ℝ), ε > 0 ∧ ∀ x, |x - x₀| < ε → f x ≤ f x₀) ∨
  (∃ (ε : ℝ), ε > 0 ∧ ∀ x, |x - x₀| < ε → f x ≥ f x₀) →
  f_derivative x₀ = 0 →
  (1 + x₀^2) * (1 + Real.cos (2 * x₀)) = 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_property_l241_24171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_minimum_sum_l241_24150

/-- An arithmetic sequence with first term a₁ and common difference d -/
noncomputable def arithmeticSequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1 : ℝ)

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def arithmeticSum (a₁ d : ℝ) (n : ℕ) : ℝ := n * (a₁ + arithmeticSequence a₁ d n) / 2

theorem arithmetic_sequence_minimum_sum :
  ∃ (a₁ d : ℝ),
    a₁ = -11 ∧
    arithmeticSequence a₁ d 4 + arithmeticSequence a₁ d 6 = -6 ∧
    (∀ n : ℕ, n > 0 → arithmeticSum a₁ d 6 ≤ arithmeticSum a₁ d n) :=
by sorry

#check arithmetic_sequence_minimum_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_minimum_sum_l241_24150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_expressions_l241_24158

theorem calculate_expressions :
  (Real.sqrt 12 + |1 - Real.sqrt 3| - (Real.pi - 3)^0 = 3 * Real.sqrt 3 - 2) ∧
  ((Real.sqrt 18 + 2) / Real.sqrt 2 - (Real.sqrt 2 - 1)^2 = 3 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_expressions_l241_24158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vann_cats_cleaned_l241_24181

/-- Represents the number of teeth for each animal type --/
structure AnimalTeeth where
  dog : Nat
  cat : Nat
  pig : Nat

/-- Represents the number of animals Vann will clean --/
structure AnimalCount where
  dogs : Nat
  cats : Nat
  pigs : Nat

/-- Calculates the total number of teeth cleaned --/
def totalTeethCleaned (teeth : AnimalTeeth) (count : AnimalCount) : Nat :=
  teeth.dog * count.dogs + teeth.cat * count.cats + teeth.pig * count.pigs

theorem vann_cats_cleaned (teeth : AnimalTeeth) (count : AnimalCount) :
  teeth.dog = 42 →
  teeth.cat = 30 →
  teeth.pig = 28 →
  count.dogs = 5 →
  count.pigs = 7 →
  totalTeethCleaned teeth count = 706 →
  count.cats = 10 := by
  sorry

-- Remove the #eval statement as it's causing issues
-- #eval vann_cats_cleaned
--   { dog := 42, cat := 30, pig := 28 }
--   { dogs := 5, cats := 10, pigs := 7 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vann_cats_cleaned_l241_24181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dune_buggy_average_speed_l241_24190

/-- The average speed of a dune buggy given different terrain speeds and equal time spent on each terrain -/
theorem dune_buggy_average_speed (flat_speed downhill_speed_increase uphill_speed_decrease : ℝ) :
  flat_speed = 60 →
  downhill_speed_increase = 12 →
  uphill_speed_decrease = 18 →
  let downhill_speed := flat_speed + downhill_speed_increase
  let uphill_speed := flat_speed - uphill_speed_decrease
  let harmonic_mean := 3 / (1 / flat_speed + 1 / downhill_speed + 1 / uphill_speed)
  abs (harmonic_mean - 55.19) < 0.01 := by
  intro h1 h2 h3
  -- The proof goes here
  sorry

#check dune_buggy_average_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dune_buggy_average_speed_l241_24190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_exponential_equation_l241_24107

theorem unique_solution_exponential_equation :
  ∃! x : ℝ, (10 : ℝ)^x * (100 : ℝ)^(2*x) = (1000 : ℝ)^5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_exponential_equation_l241_24107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l241_24119

def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℤ := {x : ℤ | 0 ≤ x ∧ x < 3}

theorem intersection_A_B : A ∩ B = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l241_24119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_calculations_l241_24182

theorem sqrt_calculations :
  ((Real.sqrt 50 - Real.sqrt 8) / Real.sqrt 2 = 3) ∧
  (Real.sqrt (3/4) * Real.sqrt (8/3) = Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_calculations_l241_24182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_factorial_inequality_l241_24135

/-- Nested factorial function -/
def nestedFactorial (n : ℕ) : ℕ → ℕ
  | 0 => n
  | k + 1 => Nat.factorial (nestedFactorial n k)

/-- Theorem stating that for all positive integers k, 
    the k-th nested factorial of 4 is less than or equal to 
    the (k+1)-th nested factorial of 3 -/
theorem nested_factorial_inequality : 
  ∀ k : ℕ, k > 0 → nestedFactorial 4 k ≤ nestedFactorial 3 (k + 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_factorial_inequality_l241_24135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l241_24139

theorem tan_alpha_value (α : ℝ) (h1 : α ∈ Set.Ioo π (3*π/2)) 
  (h2 : Real.sin α = -(Real.sqrt 5) / 5) : Real.tan α = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l241_24139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_john_and_david_l241_24127

/-- The number of workers in the hospital -/
def total_workers : ℕ := 10

/-- The number of workers to be chosen for the interview -/
def chosen_workers : ℕ := 2

/-- The probability of selecting both John and David -/
def prob_john_and_david : ℚ := 1 / 45

/-- Theorem stating that the probability of selecting both John and David
    when randomly choosing 2 workers out of 10 is equal to 1/45 -/
theorem probability_john_and_david :
  (1 : ℚ) / (Nat.choose total_workers chosen_workers) = prob_john_and_david := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_john_and_david_l241_24127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_magnitude_l241_24183

def a : ℝ × ℝ := (1, 2)
def b (l : ℝ) : ℝ × ℝ := (l, -1)

theorem perpendicular_vectors_magnitude (l : ℝ) :
  (a.1 * (b l).1 + a.2 * (b l).2 = 0) →
  ‖(a.1 + (b l).1, a.2 + (b l).2)‖ = Real.sqrt 10 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_magnitude_l241_24183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_length_of_T_l241_24176

/-- The set T of points (x, y) in the Cartesian plane satisfying the given equation -/
def T : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |abs (abs p.1 - 3) - 2| + |abs (abs p.2 - 3) - 2| = 2}

/-- The total length of all lines that make up the set T -/
noncomputable def total_length (S : Set (ℝ × ℝ)) : ℝ := sorry

/-- Theorem stating that the total length of lines in T is 128 -/
theorem total_length_of_T : total_length T = 128 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_length_of_T_l241_24176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_equals_negative_fifteen_fourths_l241_24121

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 3 + 3 * x
  else x^2 - 3 * x

-- State the theorem
theorem f_composition_equals_negative_fifteen_fourths :
  f (f (3/2)) = -15/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_equals_negative_fifteen_fourths_l241_24121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_polynomials_without_roots_l241_24188

/-- Represents a quadratic polynomial of the form x^2 + a*x + b -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ

/-- Represents a sequence of 9 quadratic polynomials -/
def PolynomialSequence := Fin 9 → QuadraticPolynomial

/-- Checks if a sequence forms an arithmetic progression -/
def isArithmeticProgression (s : Fin 9 → ℝ) : Prop :=
  ∀ i j k : Fin 9, i.val + k.val = 2 * j.val → s i + s k = 2 * s j

/-- The sum of all polynomials in the sequence -/
def sumPolynomials (ps : PolynomialSequence) (x : ℝ) : ℝ :=
  (Finset.sum Finset.univ fun i => x^2 + (ps i).a * x + (ps i).b)

theorem max_polynomials_without_roots
  (ps : PolynomialSequence)
  (h_a_prog : isArithmeticProgression (fun i => (ps i).a))
  (h_b_prog : isArithmeticProgression (fun i => (ps i).b))
  (h_sum_root : ∃ x : ℝ, sumPolynomials ps x = 0) :
  ∃ (S : Finset (Fin 9)), S.card ≥ 5 ∧ ∀ i ∈ S, ∃ x : ℝ, x^2 + (ps i).a * x + (ps i).b = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_polynomials_without_roots_l241_24188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_max_at_three_l241_24147

/-- Profit function for a manufacturer's promotional event -/
noncomputable def profit (x : ℝ) : ℝ := 28 - x - 16 / (x + 1)

/-- Theorem stating that the profit function reaches its maximum when x = 3 -/
theorem profit_max_at_three :
  ∀ x : ℝ, x ≥ 0 → profit x ≤ profit 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_max_at_three_l241_24147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lisa_finishes_first_l241_24129

/-- Represents a runner in the race -/
structure Runner where
  name : String
  distance : ℚ
  speed : ℚ

/-- Calculates the time taken by a runner to complete their distance -/
def runningTime (runner : Runner) : ℚ := runner.distance / runner.speed

theorem lisa_finishes_first (sam harvey lisa : Runner)
  (h1 : sam.distance = 12)
  (h2 : harvey.distance = sam.distance + 8)
  (h3 : lisa.distance = harvey.distance - 5)
  (h4 : sam.speed = 6)
  (h5 : harvey.speed = 7)
  (h6 : lisa.speed = 8) :
  runningTime lisa < runningTime sam ∧ runningTime lisa < runningTime harvey := by
  sorry

#check lisa_finishes_first

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lisa_finishes_first_l241_24129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_power_equation_l241_24175

theorem eight_power_equation (x : ℝ) : (1 / 8 : ℝ) * (2 : ℝ)^36 = (8 : ℝ)^x → x = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_power_equation_l241_24175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_specific_complex_l241_24151

/-- The midpoint of a line segment connecting two complex points -/
noncomputable def complex_midpoint (z₁ z₂ : ℂ) : ℂ := (z₁ + z₂) / 2

/-- Theorem: The midpoint of the line segment connecting 1-3i and (1+i)(2-i) is 2-i -/
theorem midpoint_specific_complex : 
  complex_midpoint (1 - 3*I) ((1 + I) * (2 - I)) = 2 - I :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_specific_complex_l241_24151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_and_zeros_l241_24109

noncomputable def m (ω x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin (ω * x), 1 + Real.cos (ω * x))

noncomputable def n (ω x : ℝ) : ℝ × ℝ := (Real.cos (ω * x), 1 - Real.cos (ω * x))

noncomputable def f (ω x : ℝ) : ℝ := (m ω x).1 * (n ω x).1 + (m ω x).2 * (n ω x).2

theorem symmetry_and_zeros (ω : ℝ) (h1 : ω > 0) 
  (h2 : ∃ (axis : ℝ), ∃ (center : ℝ), |axis - center| = π / 4 ∧ 
    (∀ x : ℝ, f ω (2 * center - x) = f ω x)) :
  (∃ k : ℤ, ∀ x : ℝ, f ω (x + k * π / 2 + π / 12) = f ω x) ∧
  (∀ m : ℝ, (∃ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ π / 2 ∧ 
    f ω x + m = 0 ∧ f ω y + m = 0) → -3/2 < m ∧ m ≤ -1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_and_zeros_l241_24109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_circle_properties_l241_24172

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the circle (renamed to avoid conflict with builtin 'circle')
def target_circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 2

-- State the theorem
theorem ellipse_and_circle_properties :
  -- Given conditions
  let f1 : ℝ × ℝ := (-1, 0)  -- Left focus
  let f2 : ℝ × ℝ := (1, 0)   -- Right focus
  -- The distance between foci is 2
  ∃ (A B : ℝ × ℝ) (l : ℝ → ℝ),
  -- Point (1, 3/2) is on the ellipse
  ellipse_C 1 (3/2) ∧
  -- Line l passes through F₁
  l (-1) = (f1.2 : ℝ) ∧
  -- A and B are intersection points of l and C
  ellipse_C A.1 A.2 ∧ ellipse_C B.1 B.2 ∧
  A.2 = l A.1 ∧ B.2 = l B.1 ∧
  -- Area of triangle AF₂B is 12√2/7
  abs ((A.1 - 1) * B.2 - (B.1 - 1) * A.2) / 2 = 12 * Real.sqrt 2 / 7 →
  -- Conclusion
  -- The equation of the circle centered at F₂ and tangent to l is (x-1)² + y² = 2
  ∃ (x y : ℝ), l x = y ∧ target_circle x y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_circle_properties_l241_24172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_u_converges_to_1_l241_24136

noncomputable def u (n : ℕ+) : ℝ := 1 + (Real.sin n.val) / n.val

theorem u_converges_to_1 :
  ∀ ε > 0, ∃ N : ℕ+, ∀ n ≥ N, |u n - 1| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_u_converges_to_1_l241_24136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_60_minus_alpha_eq_sqrt_3_div_2_l241_24117

theorem cos_60_minus_alpha_eq_sqrt_3_div_2 (α : ℝ) :
  Real.sin (30 * π / 180 + α) = Real.sqrt 3 / 2 →
  Real.cos (60 * π / 180 - α) = Real.sqrt 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_60_minus_alpha_eq_sqrt_3_div_2_l241_24117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_in_circle_l241_24101

theorem triangle_area_in_circle (r a b c : ℝ) 
  (h1 : r = 4) 
  (h2 : a / b = 2 / 3 ∧ b / c = 3 / 4) 
  (h3 : ∃ (x : ℝ), a = 2*x ∧ b = 3*x ∧ c = 4*x) 
  (h4 : c = 2*r) : 
  (1/2) * a * b = 12 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_in_circle_l241_24101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_relation_l241_24128

/-- The eccentricity of an ellipse with semi-major axis a and semi-minor axis b -/
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

/-- Theorem: For ellipses C₁ and C₂ with given equations and eccentricity relationship, a = 2√3/3 -/
theorem ellipse_eccentricity_relation (a : ℝ) : 
  a > 1 → 
  let e₁ := eccentricity a 1
  let e₂ := eccentricity 2 1
  e₂ = Real.sqrt 3 * e₁ → 
  a = 2 * Real.sqrt 3 / 3 := by
  sorry

#check ellipse_eccentricity_relation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_relation_l241_24128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_commodity_price_changes_l241_24108

noncomputable def price_change (initial_price : ℝ) (change_percent : ℝ) : ℝ :=
  initial_price * (1 + change_percent / 100)

noncomputable def round_to_nearest_integer (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

theorem commodity_price_changes (initial_price : ℝ) (x : ℝ) 
  (h_initial_positive : initial_price > 0) : 
  let price_after_jan := price_change initial_price 30
  let price_after_feb := price_change price_after_jan 15
  let price_after_mar := price_change price_after_feb (-25)
  let final_price := price_change price_after_mar (-x)
  final_price = initial_price →
  round_to_nearest_integer x = 11 := by
  sorry

-- Remove the #eval statement as it's not necessary for the theorem
-- and may cause issues with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_commodity_price_changes_l241_24108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_wheel_revolutions_l241_24189

/-- Calculates the number of revolutions of the back wheel of a bicycle given the radii of both wheels and the number of revolutions of the front wheel, assuming no slippage. -/
theorem bicycle_wheel_revolutions 
  (front_radius : ℝ) 
  (back_radius : ℝ) 
  (front_revolutions : ℝ) 
  (h1 : front_radius = 3) 
  (h2 : back_radius = 0.5) 
  (h3 : front_revolutions = 150) : 
  (front_radius * front_revolutions) / back_radius = 900 := by
  -- Substitute the given values
  rw [h1, h2, h3]
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

#check bicycle_wheel_revolutions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_wheel_revolutions_l241_24189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_of_increased_and_decreased_value_l241_24126

theorem difference_of_increased_and_decreased_value (x : ℝ) (h : x = 80) :
  x * 1.125 - x * 0.75 = 30 := by
  rw [h]
  norm_num

#eval (80 : ℝ) * 1.125 - 80 * 0.75

end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_of_increased_and_decreased_value_l241_24126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_properties_l241_24197

-- Define the sequence a_n
def a : ℕ → ℕ → ℕ
  | r, 0 => 1  -- Add this case for n = 0
  | r, 1 => 1
  | r, (n + 2) => (n * a r (n + 1) + 2 * (n + 2)^(2 * r)) / (n + 2)

-- Theorem statement
theorem a_properties (r : ℕ) (h : r > 0) :
  (∀ n : ℕ, a r n > 0) ∧
  (∀ n : ℕ, n > 0 → (a r n % 2 = 0 ↔ n % 4 = 0 ∨ n % 4 = 3)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_properties_l241_24197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_even_l241_24114

def is_non_zero_digit (n : ℕ) : Bool :=
  ∀ d, d < 10 → (n / 10^d) % 10 ≠ 0

def product_without_zeros : ℕ :=
  (List.range 100).filter (λ n => is_non_zero_digit (n + 1)) |>.map (· + 1) |>.prod

theorem last_digit_even :
  product_without_zeros % 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_even_l241_24114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_ratio_l241_24160

/-- Given a quadratic expression x^2 + 900x + 1800, which can be written as (x+b)^2 + c,
    prove that c/b = -446.2 (repeating) -/
theorem quadratic_ratio : 
  ∃ (b c : ℝ), (∀ x, x^2 + 900*x + 1800 = (x+b)^2 + c) ∧ c/b = -446.2 := by
  -- We'll use 'sorry' to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_ratio_l241_24160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_exp_to_line_l241_24174

/-- The shortest distance from any point on the curve y=e^x to the line y=x is √2/2 -/
theorem shortest_distance_exp_to_line :
  ∃ x : ℝ, ∀ a : ℝ, 
  let P : ℝ × ℝ := (a, Real.exp a)
  let d : ℝ → ℝ := λ t ↦ |Real.exp t - t| / Real.sqrt 2
  d x ≤ d a ∧ d x = Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_exp_to_line_l241_24174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_statements_l241_24168

open Real

theorem trigonometric_statements :
  (¬ (∀ x : ℝ, sin x + 1 / sin x ≥ 2) ↔ ∃ x : ℝ, sin x + 1 / sin x < 2) ∧
  (∀ x ∈ Set.Ioo 0 (π / 2), tan x + 1 / tan x ≥ 2) ∧
  (∃ x : ℝ, sin x + cos x = sqrt 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_statements_l241_24168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_0_minus_2i_l241_24173

noncomputable def dilation (center : ℂ) (scale : ℝ) (z : ℂ) : ℂ :=
  center + scale • (z - center)

theorem dilation_0_minus_2i : 
  dilation (1 + 2*I) 4 (0 - 2*I) = -3 - 14*I :=
by
  -- Unfold the definition of dilation
  unfold dilation
  -- Simplify the expression
  simp [Complex.add_re, Complex.add_im, Complex.mul_re, Complex.mul_im]
  -- The rest of the proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_0_minus_2i_l241_24173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_100_value_l241_24153

def sequence_a : ℕ → ℕ
  | 0 => 1
  | (n + 1) => sequence_a n + 4

theorem a_100_value : sequence_a 99 = 397 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_100_value_l241_24153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jill_net_salary_l241_24122

/-- Represents Jill's monthly finances --/
structure JillFinances where
  netSalary : ℝ
  discretionaryIncome : ℝ
  vacationFund : ℝ
  savings : ℝ
  eatingOut : ℝ
  giftsCharity : ℝ
  fitnessWellness : ℝ

/-- Jill's financial conditions --/
def jillConditions (j : JillFinances) : Prop :=
  j.discretionaryIncome = j.netSalary / 5 ∧
  j.vacationFund = 0.27 * j.discretionaryIncome ∧
  j.savings = 0.18 * j.discretionaryIncome ∧
  j.eatingOut = 0.315 * j.discretionaryIncome ∧
  j.fitnessWellness = 0.10 * j.discretionaryIncome ∧
  j.giftsCharity = 102

/-- Theorem stating Jill's net monthly salary --/
theorem jill_net_salary (j : JillFinances) (h : jillConditions j) :
  ∃ (ε : ℝ), abs (j.netSalary - 3777.78) < ε ∧ ε > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jill_net_salary_l241_24122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_water_amount_l241_24178

/-- Proves that the initial amount of water in a bowl is 10 ounces, given the evaporation rate,
    period, and percentage of water evaporated. -/
theorem initial_water_amount (daily_evaporation : ℝ) (evaporation_period : ℕ) 
  (evaporation_percentage : ℝ) (h1 : daily_evaporation = 0.0008)
  (h2 : evaporation_period = 50) (h3 : evaporation_percentage = 0.004) :
  daily_evaporation * (evaporation_period : ℝ) / evaporation_percentage = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_water_amount_l241_24178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_coefficient_terms_l241_24167

/-- Represents a term in the expansion of (x - 1/x)^6 --/
structure Term where
  coefficient : ℤ
  exponent : ℕ
  deriving Repr

/-- The expansion of (x - 1/x)^6 --/
def expansion : List Term := sorry

/-- Whether a term is odd-numbered in the expansion --/
def is_odd_term (t : Term) : Prop := sorry

/-- Whether a term is even-numbered in the expansion --/
def is_even_term (t : Term) : Prop := sorry

/-- The absolute value of a term's coefficient --/
def abs_coefficient (t : Term) : ℕ := sorry

theorem largest_coefficient_terms :
  ∃ t3 t5 : Term,
    t3 ∈ expansion ∧
    t5 ∈ expansion ∧
    abs_coefficient t3 = abs_coefficient t5 ∧
    (∀ t ∈ expansion,
      (is_odd_term t → t.coefficient > 0) ∧
      (is_even_term t → t.coefficient < 0)) ∧
    (∀ t' ∈ expansion, t' ≠ t3 ∧ t' ≠ t5 →
      abs_coefficient t' ≤ abs_coefficient t3) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_coefficient_terms_l241_24167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_when_k_eq_one_no_zeros_iff_k_gt_one_l241_24112

-- Define the function f(x) with parameter k
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := Real.log (x - 1) - k * (x - 1) + 1

-- Theorem 1: Maximum value of f when k = 1
theorem max_value_when_k_eq_one :
  ∃ (M : ℝ), M = 0 ∧ ∀ x > 1, f 1 x ≤ M := by
  sorry

-- Theorem 2: Condition for f to have no zeros
theorem no_zeros_iff_k_gt_one :
  ∀ k : ℝ, (∀ x > 1, f k x ≠ 0) ↔ k > 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_when_k_eq_one_no_zeros_iff_k_gt_one_l241_24112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_arrangements_count_l241_24161

-- Define the type for ice cream flavors
inductive Flavor
  | vanilla
  | chocolate
  | strawberry
  | cherry
  | mint

-- Define the type for a stack of ice cream scoops
def IceCreamStack := List Flavor

-- Function to count valid arrangements
def countValidArrangements : ℕ :=
  let totalFlavors := 5
  (totalFlavors - 1).factorial

-- Theorem statement
theorem valid_arrangements_count :
  countValidArrangements = 24 := by
  -- Unfold the definition of countValidArrangements
  unfold countValidArrangements
  -- Simplify the expression
  simp
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_arrangements_count_l241_24161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_y_l241_24137

-- Define the greatest integer function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := ↑(floor x)

-- State the theorem
theorem range_of_y (x : ℝ) (h : x ∈ Set.Ioc (-2.5) 3) :
  x - f x ∈ Set.Ico 0 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_y_l241_24137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_remainder_l241_24157

noncomputable def ω : ℂ := (1 + Complex.I * Real.sqrt 3) / 2

def f (a b c : ℝ) (z : ℂ) : ℂ := a * z^2018 + b * z^2017 + c * z^2016

theorem polynomial_remainder 
  (a b c : ℝ) 
  (h1 : a ≤ 2019) (h2 : b ≤ 2019) (h3 : c ≤ 2019)
  (h4 : f a b c ω = 2015 + 2019 * Complex.I * Real.sqrt 3) :
  ∃ k : ℤ, f a b c 1 = 1000 * k + 53 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_remainder_l241_24157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l241_24195

/-- Represents a digit from 1 to 9 -/
def Digit := Fin 9

/-- Represents a two-digit number -/
def TwoDigitNumber := Fin 90

/-- Represents a five-digit number -/
def FiveDigitNumber := Fin 90000

/-- Check if a number is divisible by 3 -/
def isDivisibleBy3 (n : ℕ) : Prop := ∃ k : ℕ, n = 3 * k

/-- Convert a five-digit number to its individual digits -/
def toDigits (n : FiveDigitNumber) : Fin 5 → Digit :=
  sorry

/-- Check if all digits in a five-digit number are unique -/
def hasUniqueDigits (n : FiveDigitNumber) : Prop :=
  sorry

/-- Extract the last two digits of a five-digit number -/
def lastTwoDigits (n : FiveDigitNumber) : ℕ :=
  sorry

/-- Check if a five-digit number satisfies the equation 2016 = DE × □ × □ -/
def satisfiesEquation (n : FiveDigitNumber) : Prop :=
  sorry

theorem unique_solution :
  ∃! (n : FiveDigitNumber),
    hasUniqueDigits n ∧
    ¬isDivisibleBy3 (lastTwoDigits n) ∧
    satisfiesEquation n ∧
    n.val = 85132 :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l241_24195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_f_range_a_l241_24125

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (4*a - 1)*x + 4*a else a^x

theorem decreasing_f_range_a :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x > f a y) →
  a ∈ Set.Icc (1/7 : ℝ) (1/4 : ℝ) ∧ a ≠ 1/4 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_f_range_a_l241_24125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_base9_l241_24100

/-- Represents a number in base 9 of the form bd4f₉ -/
structure Base9Number where
  b : Nat
  d : Nat
  f : Nat
  b_nonzero : b ≠ 0
  f_in_range : f ≤ 8

/-- Converts a Base9Number to its decimal representation -/
def toDecimal (n : Base9Number) : Nat :=
  729 * n.b + 81 * n.d + 36 + n.f

theorem perfect_square_base9 (n : Base9Number) :
  ∃ (m : Nat), toDecimal n = m ^ 2 → n.f ∈ ({0, 1, 4} : Set Nat) := by
  sorry

#check perfect_square_base9

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_base9_l241_24100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_amount_to_share_l241_24143

/-- The amount Nina must give Marcel to equally share the total costs -/
noncomputable def amount_to_share (X Y Z : ℝ) : ℝ := (X + Z - Y) / 2

theorem correct_amount_to_share 
  (X Y Z : ℝ) 
  (h1 : Y > X) : 
  amount_to_share X Y Z = (X + Z + Y) / 2 - Y := by
  -- Unfold the definition of amount_to_share
  unfold amount_to_share
  -- Perform algebraic manipulations
  ring
  -- The proof is complete
  done

#check correct_amount_to_share

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_amount_to_share_l241_24143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_earthquake_magnitude_and_ratio_l241_24113

-- Define the Richter magnitude scale formula
noncomputable def richter_magnitude (A : ℝ) (A₀ : ℝ) : ℝ := Real.log A - Real.log A₀

-- Define the ratio of amplitudes for two magnitudes
noncomputable def amplitude_ratio (M₁ M₂ : ℝ) : ℝ := (10 : ℝ) ^ (M₁ - M₂)

theorem earthquake_magnitude_and_ratio :
  let A := (1000 : ℝ)
  let A₀ := (0.001 : ℝ)
  let M := richter_magnitude A A₀
  (M = 6) ∧
  (amplitude_ratio 9 5 = 10000) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_earthquake_magnitude_and_ratio_l241_24113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_semiperimeter_inequality_l241_24120

-- Define a triangle type
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  sa : ℝ
  sb : ℝ
  sc : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  pos_sa : 0 < sa
  pos_sb : 0 < sb
  pos_sc : 0 < sc
  triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b

-- Define the semiperimeter
noncomputable def semiperimeter (t : Triangle) : ℝ := (t.a + t.b + t.c) / 2

-- State the theorem
theorem median_semiperimeter_inequality (t : Triangle) :
  (3 / 2) * semiperimeter t < t.sa + t.sb + t.sc ∧ t.sa + t.sb + t.sc < 2 * semiperimeter t := by
  sorry

#check median_semiperimeter_inequality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_semiperimeter_inequality_l241_24120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_value_of_f_l241_24154

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then (2 : ℝ)^x + a else x^2 - a*x

-- State the theorem
theorem minimum_value_of_f (a : ℝ) :
  (∀ x, f a x ≥ a) ∧ (∃ x, f a x = a) → a = -4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_value_of_f_l241_24154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_condition_l241_24102

def A (a : ℝ) : Set (ℕ × ℕ) :=
  {p | p.1^2 + 2*p.2^2 < a}

theorem subset_condition (a : ℝ) (h : 1 < a ∧ a ≤ 2) :
  Finset.card (Finset.filter (fun p => p.1^2 + 2*p.2^2 < a) (Finset.product (Finset.range 2) (Finset.range 2))) = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_condition_l241_24102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l241_24191

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x + 1 / (x - 3)

-- State the theorem
theorem f_minimum_value :
  ∀ x > 3, f x ≥ 5 ∧ ∃ x₀ > 3, f x₀ = 5 :=
by
  sorry

-- Additional lemma to demonstrate the minimum point
lemma f_minimum_at_four :
  f 4 = 5 :=
by
  unfold f
  norm_num

-- Proof that 4 > 3
lemma four_gt_three : 4 > 3 :=
by norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l241_24191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_Q3_l241_24169

/-- Represents a sequence of polyhedra Qᵢ --/
def Q : ℕ → ℝ := sorry

/-- Volume of the initial tetrahedron Q₀ --/
noncomputable def initial_volume : ℝ := 8

/-- The ratio of edge lengths between consecutive polyhedra --/
noncomputable def edge_ratio : ℝ := 1 / 4

/-- The volume ratio between a new tetrahedron and its parent --/
noncomputable def volume_ratio : ℝ := edge_ratio ^ 3

/-- Number of new tetrahedra added at each step after Q₁ --/
def tetrahedra_per_step : ℕ := 4 * 6

/-- The volume of Qᵢ --/
noncomputable def volume : ℕ → ℝ
| 0 => initial_volume
| 1 => initial_volume + 4 * volume_ratio * initial_volume
| (n + 2) => volume (n + 1) + tetrahedra_per_step * volume_ratio ^ (n + 1) * initial_volume * volume_ratio

theorem volume_Q3 :
  volume 3 = 8201 / 1024 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_Q3_l241_24169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_EFGH_area_l241_24140

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Represents a rectangle defined by two opposite corners -/
structure Rectangle where
  corner1 : Point
  corner2 : Point

/-- Calculates the area of a rectangle -/
def rectangleArea (r : Rectangle) : ℝ :=
  let width := |r.corner1.x - r.corner2.x|
  let height := |r.corner1.y - r.corner2.y|
  width * height

theorem rectangle_EFGH_area :
  ∀ y : ℤ,
  let E : Point := ⟨1, -5⟩
  let F : Point := ⟨2001, 195⟩
  let H : Point := ⟨3, ↑y⟩
  let EFGH : Rectangle := ⟨E, F⟩
  rectangleArea EFGH = 40400 := by
  sorry

#check rectangle_EFGH_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_EFGH_area_l241_24140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_point_on_segment_l241_24116

-- Define a point in 2D space with integer coordinates
structure Point where
  x : ℤ
  y : ℤ

-- Define a set of 5 points
def FivePoints : Set Point := {p : Point | ∃ i : Fin 5, p ∈ Set.univ}

-- Define a function to check if a point is on a line segment
def isOnSegment (p q r : Point) : Prop :=
  ∃ t : ℚ, 0 < t ∧ t < 1 ∧ 
    r.x = ⌊(1 - t) * p.x + t * q.x⌋ ∧
    r.y = ⌊(1 - t) * p.y + t * q.y⌋

-- Theorem statement
theorem integer_point_on_segment (points : Set Point) (h : points = FivePoints) :
  ∃ (p q r : Point), p ∈ points ∧ q ∈ points ∧ p ≠ q ∧ 
    isOnSegment p q r :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_point_on_segment_l241_24116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_period_of_f_perimeter_of_triangle_l241_24138

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 4 - 2 * Real.sin (x + Real.pi / 3)

-- Define the properties of triangle ABC
structure Triangle :=
  (A B C : ℝ)
  (BC : ℝ)
  (area : ℝ)
  (h_BC : BC = 3)
  (h_area : area = 3 * Real.sqrt 3 / 4)
  (h_f_A : f A = 4)

-- Theorem 1: The smallest positive period of f is 2π
theorem smallest_period_of_f : 
  ∀ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) → T ≥ 2 * Real.pi := by
  sorry

-- Theorem 2: The perimeter of triangle ABC is 3 + 2√3
theorem perimeter_of_triangle (t : Triangle) : 
  t.A + t.B + t.C = 3 + 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_period_of_f_perimeter_of_triangle_l241_24138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_intervals_l241_24115

noncomputable def f (x : ℝ) := 2 * Real.sin (-2 * x + Real.pi / 6)

theorem f_monotone_increasing_intervals :
  ∀ k : ℤ, MonotoneOn f (Set.Icc (k * Real.pi + Real.pi / 3) (k * Real.pi + 5 * Real.pi / 6)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_intervals_l241_24115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_in_Q_l241_24162

-- Define R_k as a number consisting of k repeated digits of 1 in base-ten
def R (k : ℕ) : ℕ := (10^k - 1) / 9

-- Define the quotient Q
def Q : ℕ := R 30 / R 6

-- Function to count zeros in the base-ten representation of a natural number
def countZeros (n : ℕ) : ℕ :=
  (n.repr.toList.filter (· = '0')).length

-- Theorem statement
theorem zeros_in_Q : countZeros Q = 25 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_in_Q_l241_24162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_sum_and_coefficient_sum_l241_24156

/-- Given two polynomials in d, prove that their sum simplifies to a specific form and that the sum of the coefficients equals 53. -/
theorem polynomial_sum_and_coefficient_sum :
  ∃ (d : ℚ), 
  let p1 : Polynomial ℚ := 15 * X + 17 + 16 * X^2
  let p2 : Polynomial ℚ := 3 * X + 2
  let sum : Polynomial ℚ := p1 + p2
  (sum = 16 * X^2 + 18 * X + 19) ∧ 
  (16 + 18 + 19 = 53) :=
by
  use 0  -- We can use any value for d since the theorem doesn't depend on its specific value
  -- Define the polynomials
  let p1 : Polynomial ℚ := 15 * X + 17 + 16 * X^2
  let p2 : Polynomial ℚ := 3 * X + 2
  let sum : Polynomial ℚ := p1 + p2
  
  -- Prove the two parts of the conjunction
  constructor
  
  -- Part 1: Prove that the sum simplifies to the correct form
  · sorry  -- This part requires actual computation and simplification

  -- Part 2: Prove that 16 + 18 + 19 = 53
  · norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_sum_and_coefficient_sum_l241_24156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_of_hyperbola_l241_24185

/-- The equation of the hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  2 * x^2 - 3 * y^2 + 6 * x - 12 * y - 8 = 0

/-- The coordinates of one focus of the hyperbola -/
noncomputable def focus_coordinates : ℝ × ℝ :=
  (5 * Real.sqrt 3 / 6 - 3 / 2, -2)

/-- Theorem stating that the given coordinates are a focus of the hyperbola -/
theorem focus_of_hyperbola :
  let (x, y) := focus_coordinates
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    c^2 = a^2 + b^2 ∧
    ∀ (x' y' : ℝ), hyperbola_equation x' y' ↔
      ((x' - x + 3/2)^2 / a^2) - ((y' - y)^2 / b^2) = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_of_hyperbola_l241_24185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_wall_top_value_l241_24118

/-- Represents a row in the number wall -/
def Row := List Int

/-- Checks if a row is valid according to the number wall rules -/
def isValidRow (lower upper : Row) : Prop :=
  upper.length + 1 = lower.length ∧
  ∀ i, i < upper.length → upper.get! i = lower.get! i + lower.get! (i+1)

/-- The number wall structure -/
structure NumberWall where
  row1 : Row
  row2 : Row
  row3 : Row
  row4 : Row
  valid2 : isValidRow row1 row2
  valid3 : isValidRow row2 row3
  valid4 : isValidRow row3 row4

/-- The theorem statement -/
theorem number_wall_top_value (wall : NumberWall)
  (h1 : wall.row1 = [4, 5, 12, 13])
  (h2 : wall.row2 = [7, 16, 25])
  (h3 : wall.row3 = [2, 29])
  (h4 : wall.row4.length = 1) :
  wall.row4.head! = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_wall_top_value_l241_24118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interval_of_decrease_l241_24199

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := (1/2)^x

-- Define the property of f being symmetric to g with respect to y = x
def symmetric_to_g (f : ℝ → ℝ) : Prop :=
  ∀ x y, f x = y ↔ g y = x

-- Define the inner function h
def h (x : ℝ) : ℝ := 2*x - x^2

-- State the theorem
theorem interval_of_decrease (f : ℝ → ℝ) (h_sym : symmetric_to_g f) :
  ∃ a b, a = 0 ∧ b = 1 ∧ 
  ∀ x₁ x₂, a < x₁ ∧ x₁ < x₂ ∧ x₂ < b → f (h x₂) < f (h x₁) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interval_of_decrease_l241_24199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_properties_l241_24186

/-- Complex number z as a function of real number m -/
def z (m : ℝ) : ℂ := Complex.mk (m^2 - 2*m - 3) (m^2 + 3*m + 2)

/-- z is a pure imaginary number -/
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

/-- z is in the second quadrant -/
def is_in_second_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im > 0

theorem z_properties (m : ℝ) :
  (is_pure_imaginary (z m) ↔ m = 3) ∧
  (is_in_second_quadrant (z m) ↔ -1 < m ∧ m < 3) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_properties_l241_24186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_identification_l241_24130

noncomputable section

def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

def f (x : ℝ) : ℝ := 3 * x^2 + x - 1
def g (x : ℝ) : ℝ := 3 * x - 1
def h (x : ℝ) : ℝ := 1 / x^2
def k (x : ℝ) : ℝ := 2 * x^3 - 1

theorem quadratic_function_identification :
  is_quadratic f ∧ ¬is_quadratic g ∧ ¬is_quadratic h ∧ ¬is_quadratic k :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_identification_l241_24130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_A_l241_24177

def A : Matrix (Fin 2) (Fin 2) ℚ := !![1, 4; 2, 5]

noncomputable def inverse_or_zero (M : Matrix (Fin 2) (Fin 2) ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  if M.det ≠ 0 then M⁻¹ else 0

theorem inverse_of_A :
  inverse_or_zero A = !![(-5:ℚ)/3, 4/3; 2/3, (-1:ℚ)/3] := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_A_l241_24177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l241_24179

/-- The speed of a train in km/hr given its length and time to cross a point -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  (length / time) * 3.6

/-- Theorem: A train 100 m long crossing an electric pole in 4.99960003199744 seconds 
    has a speed of approximately 72.01 km/hr -/
theorem train_speed_calculation :
  let length : ℝ := 100
  let time : ℝ := 4.99960003199744
  abs (train_speed length time - 72.01) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l241_24179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_x_coordinate_l241_24166

/-- Represents a point on a parabola -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ

/-- The parabola equation y² = 12x -/
def on_parabola (p : ParabolaPoint) : Prop :=
  p.y^2 = 12 * p.x

/-- The focus of the parabola y² = 12x -/
def focus : ParabolaPoint :=
  { x := 3, y := 0 }

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ParabolaPoint) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: For a point M on the parabola y² = 12x with distance 8 from the focus,
    the x-coordinate of M is 5 -/
theorem parabola_point_x_coordinate 
  (M : ParabolaPoint) 
  (h1 : on_parabola M) 
  (h2 : distance M focus = 8) : 
  M.x = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_x_coordinate_l241_24166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_neg_pi_sixth_f_at_2theta_plus_pi_third_l241_24159

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := sqrt 2 * cos (x - π / 12)

-- Theorem 1
theorem f_at_neg_pi_sixth : f (-π / 6) = 1 := by sorry

-- Theorem 2
theorem f_at_2theta_plus_pi_third (θ : ℝ) 
  (h1 : cos θ = 3 / 5) 
  (h2 : θ ∈ Set.Ioo (3 * π / 2) (2 * π)) : 
  f (2 * θ + π / 3) = 17 / 25 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_neg_pi_sixth_f_at_2theta_plus_pi_third_l241_24159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hannah_running_time_difference_l241_24141

/-- Represents Hannah's running information for a single day -/
structure RunInfo where
  distance : ℝ  -- in kilometers
  pace : ℝ      -- in minutes per kilometer

/-- Calculates the total running time given RunInfo -/
def runningTime (info : RunInfo) : ℝ :=
  info.distance * info.pace

theorem hannah_running_time_difference :
  let monday : RunInfo := { distance := 9, pace := 6 }
  let wednesday : RunInfo := { distance := 4.816, pace := 5.5 }
  let friday : RunInfo := { distance := 2.095, pace := 7 }
  let monday_time := runningTime monday
  let wednesday_friday_time := runningTime wednesday + runningTime friday
  |monday_time - wednesday_friday_time - 12.847| < 0.001 := by
  sorry

#eval let monday : RunInfo := { distance := 9, pace := 6 }
      let wednesday : RunInfo := { distance := 4.816, pace := 5.5 }
      let friday : RunInfo := { distance := 2.095, pace := 7 }
      let monday_time := runningTime monday
      let wednesday_friday_time := runningTime wednesday + runningTime friday
      monday_time - wednesday_friday_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hannah_running_time_difference_l241_24141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_number_for_max_sum_l241_24192

def is_valid_digit (d : ℕ) : Prop := d ≥ 0 ∧ d ≤ 9

def is_four_digit_number (n : ℕ) : Prop := n ≥ 1000 ∧ n ≤ 9999

def digits_used_once (a b c : ℕ) : Prop :=
  ∃ (d₁ d₂ d₃ d₄ d₅ d₆ d₇ d₈ d₉ d₁₀ d₁₁ d₁₂ : ℕ),
    is_valid_digit d₁ ∧ is_valid_digit d₂ ∧ is_valid_digit d₃ ∧ is_valid_digit d₄ ∧
    is_valid_digit d₅ ∧ is_valid_digit d₆ ∧ is_valid_digit d₇ ∧ is_valid_digit d₈ ∧
    is_valid_digit d₉ ∧ is_valid_digit d₁₀ ∧ is_valid_digit d₁₁ ∧ is_valid_digit d₁₂ ∧
    a = d₁ * 1000 + d₂ * 100 + d₃ * 10 + d₄ ∧
    b = d₅ * 1000 + d₆ * 100 + d₇ * 10 + d₈ ∧
    c = d₉ * 1000 + d₁₀ * 100 + d₁₁ * 10 + d₁₂ ∧
    Finset.card (Finset.range 10) = Finset.card {d₁, d₂, d₃, d₄, d₅, d₆, d₇, d₈, d₉, d₁₀, d₁₁, d₁₂}

def maximizes_sum (a b c : ℕ) : Prop :=
  ∀ (x y z : ℕ), 
    is_four_digit_number x ∧ is_four_digit_number y ∧ is_four_digit_number z ∧
    digits_used_once x y z →
    a + b + c ≥ x + y + z

theorem valid_number_for_max_sum :
  ∃ (b c : ℕ), 
    is_four_digit_number 9762 ∧
    is_four_digit_number b ∧
    is_four_digit_number c ∧
    digits_used_once 9762 b c ∧
    maximizes_sum 9762 b c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_number_for_max_sum_l241_24192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_construct_parallel_perpendicular_l241_24165

-- Define the necessary types and structures
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

structure Circle where
  center : Point
  radius : ℝ

-- Define the given conditions
noncomputable def A : Point := ⟨0, 0⟩
noncomputable def B : Point := ⟨1, 0⟩
noncomputable def O : Point := ⟨0.5, 0⟩
noncomputable def P : Point := ⟨0, 1⟩

-- Define the line AB and circle
noncomputable def lineAB : Line := ⟨0, 1, 0⟩
noncomputable def circleO : Circle := ⟨O, 0.5⟩

-- State the properties of the given conditions
axiom O_midpoint : O = ⟨(A.x + B.x) / 2, (A.y + B.y) / 2⟩
axiom P_not_on_lineAB : ¬ (P.x * lineAB.a + P.y * lineAB.b + lineAB.c = 0)
axiom circle_center_O : circleO.center = O
axiom circle_radius_OA : circleO.radius = Real.sqrt ((A.x - O.x)^2 + (A.y - O.y)^2)
axiom P_not_on_circle : (P.x - O.x)^2 + (P.y - O.y)^2 ≠ circleO.radius^2

-- Define the parallel and perpendicular properties
def parallel (l1 l2 : Line) : Prop := l1.a * l2.b = l1.b * l2.a
def perpendicular (l1 l2 : Line) : Prop := l1.a * l2.a + l1.b * l2.b = 0

-- State the theorem to be proved
theorem construct_parallel_perpendicular :
  ∃ (l1 l2 : Line),
    (P.x * l1.a + P.y * l1.b + l1.c = 0) ∧
    (P.x * l2.a + P.y * l2.b + l2.c = 0) ∧
    parallel l1 lineAB ∧
    perpendicular l2 lineAB :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_construct_parallel_perpendicular_l241_24165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_angle_is_60_degrees_l241_24103

noncomputable def angle_with_x_axis (slope : ℝ) : ℝ := Real.arctan slope

noncomputable def line_l : ℝ → ℝ × ℝ := fun t ↦ (t - 3, Real.sqrt 3 * t)

noncomputable def slope_l : ℝ := (line_l 1).2 - (line_l 0).2

theorem line_angle_is_60_degrees :
  angle_with_x_axis slope_l = π / 3 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_angle_is_60_degrees_l241_24103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_from_curve_to_line_l241_24124

-- Define the curve C: (x-1)² + (y-1)² = 2
def on_curve (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 2

-- Define the line l: x + y = 4
def on_line (x y : ℝ) : Prop := x + y = 4

-- Define the distance function from a point (x, y) to the line
noncomputable def distance_to_line (x y : ℝ) : ℝ := |x + y - 4| / Real.sqrt 2

-- Theorem statement
theorem max_distance_from_curve_to_line :
  ∃ (d : ℝ), d = 2 * Real.sqrt 2 ∧
  (∀ (x y : ℝ), on_curve x y → distance_to_line x y ≤ d) ∧
  (∃ (x y : ℝ), on_curve x y ∧ distance_to_line x y = d) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_from_curve_to_line_l241_24124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_l241_24152

/-- The distance between two points in 3D space -/
noncomputable def distance (x1 y1 z1 x2 y2 z2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)

/-- Point A is equidistant from points B and C -/
theorem equidistant_point :
  let A : ℝ × ℝ × ℝ := (0, 0, 1)
  let B : ℝ × ℝ × ℝ := (-18, 1, 0)
  let C : ℝ × ℝ × ℝ := (15, -10, 2)
  distance A.1 A.2.1 A.2.2 B.1 B.2.1 B.2.2 = distance A.1 A.2.1 A.2.2 C.1 C.2.1 C.2.2 :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_l241_24152
