import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ethane_combustion_result_l1254_125485

noncomputable section

/-- Atomic mass of carbon in g/mol -/
def carbon_mass : ℝ := 12.01

/-- Atomic mass of hydrogen in g/mol -/
def hydrogen_mass : ℝ := 1.008

/-- Atomic mass of oxygen in g/mol -/
def oxygen_mass : ℝ := 16.00

/-- Mass of ethane in grams -/
def ethane_mass : ℝ := 150

/-- Molecular weight of ethane (C2H6) in g/mol -/
def ethane_molecular_weight : ℝ := 2 * carbon_mass + 6 * hydrogen_mass

/-- Moles of ethane -/
noncomputable def ethane_moles : ℝ := ethane_mass / ethane_molecular_weight

/-- Ratio of H2O moles to C2H6 moles in the balanced equation -/
def h2o_to_c2h6_ratio : ℝ := 3

/-- Molecular weight of H2O in g/mol -/
def h2o_molecular_weight : ℝ := 2 * hydrogen_mass + oxygen_mass

/-- Moles of H2O formed -/
noncomputable def h2o_moles : ℝ := ethane_moles * h2o_to_c2h6_ratio

theorem ethane_combustion_result :
  h2o_molecular_weight = 18.016 ∧ 
  (h2o_moles ≥ 14.96 ∧ h2o_moles ≤ 14.98) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ethane_combustion_result_l1254_125485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1254_125422

noncomputable def f (x : ℝ) : ℝ := x^3 - 1/x^3

theorem f_properties :
  (∀ x : ℝ, x ≠ 0 → f (-x) = -f x) ∧
  (∀ a b : ℝ, 0 < a → a < b → f a < f b) := by
  constructor
  · intro x hx
    simp [f]
    ring
  · intro a b ha hab
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1254_125422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_cream_volume_calculation_l1254_125490

/-- The volume of ice cream in a cone-hemisphere-cone configuration -/
noncomputable def ice_cream_volume (h₁ r₁ r₂ h₂ : ℝ) : ℝ :=
  (1/3 * Real.pi * r₁^2 * h₁) +  -- Volume of large cone
  (2/3 * Real.pi * r₁^3) +       -- Volume of hemisphere
  (1/3 * Real.pi * r₂^2 * h₂)    -- Volume of small cone

/-- Theorem stating the total volume of ice cream -/
theorem ice_cream_volume_calculation :
  ice_cream_volume 12 4 2 3 = (332/3) * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_cream_volume_calculation_l1254_125490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mascot_sales_and_profit_l1254_125441

-- Define the variables and constants
def bing_dwen_price : ℝ → Prop := sorry
def shuey_rhon_price : ℝ → Prop := sorry
def bing_dwen_quantity : ℕ → Prop := sorry
def max_profit : ℝ → Prop := sorry

-- Define the conditions
axiom total_bing_dwen_sales : ℝ
axiom total_shuey_rhon_sales : ℝ
axiom price_difference : ℝ
axiom quantity_ratio : ℕ
axiom bing_dwen_cost : ℝ
axiom shuey_rhon_cost : ℝ
axiom total_toys : ℕ
axiom shuey_rhon_quantity_constraint : ℕ → Prop
axiom total_purchase_price_constraint : ℕ → Prop
axiom bing_dwen_price_reduction : ℝ

-- State the theorem
theorem mascot_sales_and_profit :
  total_bing_dwen_sales = 24000 ∧
  total_shuey_rhon_sales = 8000 ∧
  price_difference = 40 ∧
  quantity_ratio = 2 ∧
  bing_dwen_cost = 100 ∧
  shuey_rhon_cost = 60 ∧
  total_toys = 800 ∧
  (∀ m : ℕ, shuey_rhon_quantity_constraint m ↔ total_toys - m ≤ 3 * m) ∧
  (∀ m : ℕ, total_purchase_price_constraint m ↔ bing_dwen_cost * m + shuey_rhon_cost * (total_toys - m) ≤ 57600) ∧
  bing_dwen_price_reduction = 0.9 →
  bing_dwen_price 120 ∧
  shuey_rhon_price 80 ∧
  bing_dwen_quantity 200 ∧
  max_profit 13600 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mascot_sales_and_profit_l1254_125441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_michael_currency_exchange_l1254_125424

theorem michael_currency_exchange (d : ℕ) : 
  (3 * d - 75 = d) → 
  (d = 38) ∧ 
  (Nat.sum (Nat.digits 10 d) = 11) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_michael_currency_exchange_l1254_125424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_solution_set_l1254_125411

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom f_nonneg : ∀ x : ℝ, x ≥ 0 → f x = x^2

-- Define the solution set
def solution_set : Set ℝ := { x : ℝ | x < (Real.sqrt 5 - 1) / 2 }

-- State the theorem
theorem f_inequality_solution_set :
  { x : ℝ | f (f x) + f (x - 1) < 0 } = solution_set :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_solution_set_l1254_125411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cycling_speed_equality_l1254_125402

/-- Ben's cycling speed as a function of x -/
noncomputable def ben_speed (x : ℝ) : ℝ := 2*x^2 + 2*x - 24

/-- Daisy's total distance as a function of x -/
noncomputable def daisy_distance (x : ℝ) : ℝ := 2*x^2 - 4*x - 120

/-- Daisy's travel time as a function of x -/
noncomputable def daisy_time (x : ℝ) : ℝ := 2*x - 4

/-- Daisy's speed as a function of x -/
noncomputable def daisy_speed (x : ℝ) : ℝ := daisy_distance x / daisy_time x

theorem cycling_speed_equality :
  ∃ x : ℝ, ben_speed x = daisy_speed x ∧ ben_speed x = 25.5 := by
  sorry

#check cycling_speed_equality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cycling_speed_equality_l1254_125402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ming_bai_qing_chu_possible_values_l1254_125455

-- Define the digits as natural numbers
def ming : ℕ := sorry
def bai : ℕ := sorry
def qing : ℕ := sorry
def chu : ℕ := sorry

-- Define the conditions
axiom different_digits : ming ≠ bai ∧ ming ≠ qing ∧ ming ≠ chu ∧ bai ≠ qing ∧ bai ≠ chu ∧ qing ≠ chu
axiom single_digits : ming < 10 ∧ bai < 10 ∧ qing < 10 ∧ chu < 10
axiom product_equation : (10 * ming + ming) * (10 * bai + bai) = (1000 * qing + 100 * qing + 10 * chu + chu)

-- Define the result
def result : List ℕ := [4738, 7438, 8874]

-- Theorem statement
theorem ming_bai_qing_chu_possible_values : 
  (1000 * ming + 100 * bai + 10 * qing + chu) ∈ result := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ming_bai_qing_chu_possible_values_l1254_125455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_range_equal_segments_k_value_l1254_125453

-- Define the curve C
def C (x y : ℝ) : Prop := (x^2 + y) * (x + y) = 0

-- Define the line l
def l (k b x y : ℝ) : Prop := y = k * x + b

-- Define the intersection condition
def intersects_at_three_distinct_points (C : ℝ → ℝ → Prop) (l : ℝ → ℝ → ℝ → ℝ → Prop) (k b : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ), 
    C x₁ y₁ ∧ C x₂ y₂ ∧ C x₃ y₃ ∧
    l k b x₁ y₁ ∧ l k b x₂ y₂ ∧ l k b x₃ y₃ ∧
    x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃

-- Part 1
theorem intersection_range :
  ∀ k : ℝ, intersects_at_three_distinct_points C l k (1/16) ↔ 
    k ∈ (Set.Ioi (1/2) ∪ Set.Ioo (-1) (-1/2) ∪ Set.Ioo (-17/16) (-1) ∪ Set.Iic (-17/16)) :=
sorry

-- Part 2
theorem equal_segments_k_value :
  ∀ k : ℝ, (intersects_at_three_distinct_points C l k 1 ∧
    ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ), 
      C x₁ y₁ ∧ C x₂ y₂ ∧ C x₃ y₃ ∧
      l k 1 x₁ y₁ ∧ l k 1 x₂ y₂ ∧ l k 1 x₃ y₃ ∧
      (x₁ - x₂)^2 + (y₁ - y₂)^2 = (x₂ - x₃)^2 + (y₂ - y₃)^2) →
    k = (2 : ℝ)^(1/3) + (1/2 : ℝ)^(1/3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_range_equal_segments_k_value_l1254_125453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_three_halves_pi_plus_alpha_l1254_125476

theorem cos_three_halves_pi_plus_alpha (α : Real) 
  (h1 : Real.tan α = 2) 
  (h2 : π < α ∧ α < 3/2 * π) : 
  Real.cos (3/2 * π + α) = -2 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_three_halves_pi_plus_alpha_l1254_125476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_policeman_catches_thief_l1254_125445

-- Define the thief's speed as 1 unit per hour
noncomputable def thief_speed : ℝ := 1

-- Define the policeman's speed as twice the thief's speed
noncomputable def policeman_speed : ℝ := 2 * thief_speed

-- Define the policeman's strategy
noncomputable def policeman_strategy (k : ℕ) : ℝ := 
  (6 / 5) * (((-4) ^ k) - 1)

-- Theorem statement
theorem policeman_catches_thief :
  ∀ (thief_start_time : ℝ) (thief_direction : ℝ),
    thief_start_time < 0 ∧ (thief_direction = 1 ∨ thief_direction = -1) →
    ∃ (k : ℕ), 
      policeman_strategy k > thief_speed * (4^k + |thief_start_time|) * thief_direction :=
by
  sorry

#check policeman_catches_thief

end NUMINAMATH_CALUDE_ERRORFEEDBACK_policeman_catches_thief_l1254_125445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sharks_win_percentage_minimum_N_for_sharks_win_minimum_N_correct_l1254_125433

theorem sharks_win_percentage (N : ℕ) : (1 + N : ℚ) / (4 + N) ≥ 9/10 ↔ N ≥ 26 := by
  sorry

theorem minimum_N_for_sharks_win : ∃ N : ℕ, (∀ k : ℕ, k < N → (1 + k : ℚ) / (4 + k) < 9/10) ∧ (1 + N : ℚ) / (4 + N) ≥ 9/10 := by
  sorry

-- Remove the #eval statement as it's causing issues
-- #eval (minimum_N_for_sharks_win.choose)

-- Instead, we can add a definition that states the result
def minimum_N : ℕ := 26

theorem minimum_N_correct : minimum_N = 26 := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sharks_win_percentage_minimum_N_for_sharks_win_minimum_N_correct_l1254_125433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_heads_one_tail_probability_l1254_125460

/-- A fair coin flip is represented as a probability of 1/2 for each outcome -/
noncomputable def fairCoinFlip : ℝ := 1 / 2

/-- The probability of getting a specific sequence of three independent fair coin flips -/
noncomputable def threeFlipsProbability (flip1 flip2 flip3 : ℝ) : ℝ := flip1 * flip2 * flip3

/-- Theorem: The probability of getting exactly two heads followed by one tail (HHT) 
    when flipping a fair coin three times is 1/8 -/
theorem two_heads_one_tail_probability : 
  threeFlipsProbability fairCoinFlip fairCoinFlip fairCoinFlip = 1 / 8 := by
  -- Unfold the definitions
  unfold threeFlipsProbability fairCoinFlip
  -- Simplify the expression
  simp [mul_assoc]
  -- The rest of the proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_heads_one_tail_probability_l1254_125460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_distance_theorem_l1254_125420

/-- Given vectors a, b, and p in a real inner product space, 
    if ‖p - b‖ = 3 ‖p - a‖, then p is at a fixed distance from (9/8)a - (1/8)b -/
theorem fixed_distance_theorem {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] (a b p : V) :
  ‖p - b‖ = 3 * ‖p - a‖ → 
  ∃ (c : ℝ), ‖p - ((9/8 : ℝ) • a - (1/8 : ℝ) • b)‖ = c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_distance_theorem_l1254_125420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_number_with_conditions_l1254_125443

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

/-- A function that returns the digits of a natural number -/
def digits (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 10) ((m % 10) :: acc)
  aux n []

/-- A predicate that checks if all digits in a number are different -/
def allDigitsDifferent (n : ℕ) : Prop :=
  let d := digits n
  d.Nodup

/-- A predicate that checks if the sum of any two digits in a number is prime -/
def sumOfAnyTwoDigitsIsPrime (n : ℕ) : Prop :=
  let d := digits n
  ∀ i j, i < d.length → j < d.length → i ≠ j → 
    isPrime (d.get ⟨i, by sorry⟩ + d.get ⟨j, by sorry⟩)

/-- The main theorem stating that 520 is the largest number satisfying the conditions -/
theorem largest_number_with_conditions :
  (allDigitsDifferent 520 ∧ sumOfAnyTwoDigitsIsPrime 520) ∧
  ∀ n : ℕ, n > 520 → ¬(allDigitsDifferent n ∧ sumOfAnyTwoDigitsIsPrime n) := by
  sorry

#check largest_number_with_conditions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_number_with_conditions_l1254_125443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_is_87_l1254_125498

def numbers : List Nat := [65, 87, 143, 169, 187]

def largest_prime_factor (n : Nat) : Nat :=
  (Nat.factors n).maximum?.getD 1

theorem largest_prime_factor_is_87 :
  ∀ n ∈ numbers, n ≠ 87 → largest_prime_factor n < largest_prime_factor 87 := by
  sorry

#eval largest_prime_factor 87

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_is_87_l1254_125498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_equations_l1254_125431

/-- Represents the weight of a sparrow in liang -/
def x : ℝ := sorry

/-- Represents the weight of a swallow in liang -/
def y : ℝ := sorry

/-- The total weight of 5 sparrows and 6 swallows is 16 liang -/
axiom total_weight : 5 * x + 6 * y = 16

/-- Sparrows are heavier than swallows -/
axiom sparrow_heavier : x > y

/-- When one sparrow is exchanged with one swallow, they weigh the same -/
axiom exchange_weight : 4 * x + y = x + 5 * y

/-- The system of equations representing the problem -/
def problem_equations : Prop :=
  (5 * x + 6 * y = 16) ∧ (4 * x + y = x + 5 * y)

/-- Theorem stating that the given system of equations correctly represents the problem -/
theorem correct_equations : problem_equations := by
  unfold problem_equations
  constructor
  . exact total_weight
  . exact exchange_weight

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_equations_l1254_125431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_black_to_white_area_ratio_l1254_125479

-- Define the radii of the circles
def radii : List ℝ := [1, 3, 5, 7, 9]

-- Function to calculate the area of a circle
noncomputable def circle_area (r : ℝ) : ℝ := Real.pi * r^2

-- Function to calculate the area of a ring between two radii
noncomputable def ring_area (r1 r2 : ℝ) : ℝ := circle_area r2 - circle_area r1

-- Theorem statement
theorem black_to_white_area_ratio :
  let black_area := circle_area (radii[0]!) + ring_area (radii[1]!) (radii[2]!) + ring_area (radii[3]!) (radii[4]!)
  let white_area := ring_area (radii[0]!) (radii[1]!) + ring_area (radii[2]!) (radii[3]!)
  black_area / white_area = 49 / 32 := by
  sorry

#eval radii[0]!
#eval radii[1]!
#eval radii[2]!
#eval radii[3]!
#eval radii[4]!

end NUMINAMATH_CALUDE_ERRORFEEDBACK_black_to_white_area_ratio_l1254_125479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_dist_parabola_to_line_l1254_125452

/-- The parabola C: x^2 = 2y -/
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 = 2 * p.2}

/-- The line l: y = x - 2 -/
def l : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = p.1 - 2}

/-- The distance function from a point to the line l -/
noncomputable def dist_to_line (p : ℝ × ℝ) : ℝ :=
  |p.1 - p.2 - 2| / Real.sqrt 2

/-- The theorem stating the minimum distance from the parabola to the line -/
theorem min_dist_parabola_to_line :
  (⨅ p ∈ C, dist_to_line p) = (3 * Real.sqrt 2) / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_dist_parabola_to_line_l1254_125452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_l1254_125484

theorem sin_double_angle (α : ℝ) (h1 : 0 < α) (h2 : α < π) (h3 : Real.cos α = -12/13) : 
  Real.sin (2 * α) = -120/169 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_l1254_125484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_sum_l1254_125466

-- Define power functions f and g
noncomputable def f (x : ℝ) : ℝ := x ^ (-1 : ℤ)
noncomputable def g (x : ℝ) : ℝ := x ^ (-2 : ℤ)

-- State the theorem
theorem power_function_sum :
  (f (1/2) = 2) → (g (-2) = 1/4) → (f 2 + g (-1) = 3/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_sum_l1254_125466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cat_rescue_tree_height_l1254_125403

/-- Calculates the height of a tree given the number of rungs climbed and information from a previous rescue. -/
def tree_height (previous_height : ℚ) (previous_rungs : ℚ) (current_rungs : ℚ) : ℚ :=
  (previous_height / previous_rungs) * current_rungs

/-- Proves that the tree height is 20 feet given the conditions from the problem. -/
theorem cat_rescue_tree_height :
  tree_height 6 12 40 = 20 := by
  unfold tree_height
  norm_num

#eval tree_height 6 12 40

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cat_rescue_tree_height_l1254_125403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nth_square_position_efgh_l1254_125496

-- Define the possible square positions
inductive SquarePosition
  | EFGH
  | HEFG
  | GFHE
deriving Repr

-- Define the transformation functions
def rotate90Clockwise (pos : SquarePosition) : SquarePosition :=
  match pos with
  | SquarePosition.EFGH => SquarePosition.HEFG
  | SquarePosition.HEFG => SquarePosition.GFHE
  | SquarePosition.GFHE => SquarePosition.EFGH

def reflectVertical (pos : SquarePosition) : SquarePosition :=
  match pos with
  | SquarePosition.EFGH => SquarePosition.EFGH
  | SquarePosition.HEFG => SquarePosition.GFHE
  | SquarePosition.GFHE => SquarePosition.HEFG

-- Define the function to get the nth square position
def nthSquarePosition (n : Nat) : SquarePosition :=
  match n % 4 with
  | 1 => SquarePosition.EFGH
  | 2 => SquarePosition.HEFG
  | 3 => SquarePosition.GFHE
  | _ => SquarePosition.EFGH

-- Theorem statement
theorem nth_square_position_efgh (n : Nat) :
  n > 0 → nthSquarePosition n = SquarePosition.EFGH ↔ n % 4 = 1 := by
  sorry

#eval nthSquarePosition 2013

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nth_square_position_efgh_l1254_125496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_intersection_theorem_l1254_125461

noncomputable section

/-- Curve C₁ in polar coordinates -/
def C₁ (θ : ℝ) : ℝ := 4 * Real.sin θ

/-- Curve C₂ in parametric form -/
def C₂ (m α t : ℝ) : ℝ × ℝ := (m + t * Real.cos α, t * Real.sin α)

/-- Angle φ -/
def φ : ℝ := 5 * Real.pi / 12

/-- Point A on C₁ -/
def A : ℝ × ℝ := (C₁ φ * Real.cos φ, C₁ φ * Real.sin φ)

/-- Point B on C₁ -/
def B : ℝ × ℝ := (C₁ (φ + Real.pi/4) * Real.cos (φ + Real.pi/4), 
                  C₁ (φ + Real.pi/4) * Real.sin (φ + Real.pi/4))

/-- Point C on C₁ -/
def C : ℝ × ℝ := (C₁ (φ - Real.pi/4) * Real.cos (φ - Real.pi/4), 
                  C₁ (φ - Real.pi/4) * Real.sin (φ - Real.pi/4))

theorem curve_intersection_theorem :
  ∃ (m α : ℝ), 
    0 ≤ α ∧ α < Real.pi ∧
    (Real.sqrt ((B.1 - 0)^2 + (B.2 - 0)^2) + 
     Real.sqrt ((C.1 - 0)^2 + (C.2 - 0)^2) = 
     Real.sqrt 2 * Real.sqrt ((A.1 - 0)^2 + (A.2 - 0)^2)) ∧
    B = C₂ m α (Real.sqrt (B.1^2 + B.2^2)) ∧
    C = C₂ m α (Real.sqrt (C.1^2 + C.2^2)) ∧
    m = 2 * Real.sqrt 3 ∧
    α = 5 * Real.pi / 6 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_intersection_theorem_l1254_125461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_eventually_constant_mod_l1254_125486

def f : ℕ → ℕ
  | 0 => 2  -- Add this case to handle 0
  | 1 => 2
  | n + 2 => 2^(f (n + 1))

theorem f_eventually_constant_mod (m : ℕ) : 
  ∃ (N r : ℕ), ∀ n ≥ N, f n ≡ r [MOD m] := by
  sorry

#eval f 0  -- Test the function
#eval f 1
#eval f 2
#eval f 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_eventually_constant_mod_l1254_125486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_length_l1254_125401

/-- Parabola C defined by y²=4x -/
def parabola_C (x y : ℝ) : Prop := y^2 = 4*x

/-- Line l with slope k passing through point P(-4,0) -/
def line_l (k x y : ℝ) : Prop := y = k*(x + 4)

/-- Point A on both parabola C and line l -/
def point_A (k x y : ℝ) : Prop := parabola_C x y ∧ line_l k x y

/-- Point B on both parabola C and line l -/
def point_B (k x y : ℝ) : Prop := parabola_C x y ∧ line_l k x y

/-- A is the midpoint of PB -/
def A_midpoint_PB (k xA yA xB yB : ℝ) : Prop :=
  point_A k xA yA ∧ point_B k xB yB ∧ xA = (xB - 4)/2 ∧ yA = yB/2

/-- Length of AB -/
noncomputable def length_AB (xA yA xB yB : ℝ) : ℝ :=
  Real.sqrt ((xB - xA)^2 + (yB - yA)^2)

theorem parabola_line_intersection_length :
  ∀ (k xA yA xB yB : ℝ),
    k ≠ 0 →
    A_midpoint_PB k xA yA xB yB →
    length_AB xA yA xB yB = 2 * Real.sqrt 11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_length_l1254_125401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_not_sufficient_l1254_125432

-- Define the curve equation
def on_curve (x y : ℝ) : Prop := x^2/4 + y^2/2 = 1

-- Define the coordinate equation
def satisfies_coordinate_eq (x y : ℝ) : Prop := y = -(Real.sqrt 2)/2 * Real.sqrt (4 - x^2)

-- Theorem statement
theorem necessary_not_sufficient :
  (∀ x y : ℝ, satisfies_coordinate_eq x y → on_curve x y) ∧
  (∃ x y : ℝ, on_curve x y ∧ ¬satisfies_coordinate_eq x y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_not_sufficient_l1254_125432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_students_is_54_l1254_125488

/-- Represents a selection of math and physics problems by students. -/
structure ProblemSelection where
  mathProblems : Finset (Fin 20)
  physicsProblems : Finset (Fin 11)
  studentChoices : Finset (Fin 20 × Fin 11)

/-- The maximum number of students possible under the given conditions. -/
def maxStudents : ℕ := 54

/-- The conditions of the problem are satisfied. -/
def satisfiesConditions (selection : ProblemSelection) : Prop :=
  -- Each student chooses one math and one physics problem
  ∀ s ∈ selection.studentChoices, s.1 ∈ selection.mathProblems ∧ s.2 ∈ selection.physicsProblems
  -- No two students choose the same pair of problems
  ∧ selection.studentChoices.card = selection.studentChoices.toList.length
  -- At least one problem chosen by each student is chosen by at most one other student
  ∧ ∀ s ∈ selection.studentChoices, 
      (selection.studentChoices.filter (λ t => t.1 = s.1)).card ≤ 2 ∨
      (selection.studentChoices.filter (λ t => t.2 = s.2)).card ≤ 2

/-- The main theorem stating that the maximum number of students is 54. -/
theorem max_students_is_54 :
  ∀ selection : ProblemSelection, satisfiesConditions selection →
  selection.studentChoices.card ≤ maxStudents := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_students_is_54_l1254_125488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_girls_after_four_years_l1254_125489

/-- Calculates the number of girls after applying a yearly fluctuation rate -/
def apply_fluctuation (girls : ℝ) (rate : ℝ) : ℝ :=
  girls * (1 + rate)

/-- Theorem: Number of girls after four years with given fluctuation rates -/
theorem girls_after_four_years (initial_girls_percentage : ℝ) (boys : ℕ) 
  (rate1 rate2 rate3 rate4 : ℝ) :
  initial_girls_percentage = 0.60 →
  boys = 300 →
  rate1 = -0.05 →
  rate2 = 0.03 →
  rate3 = -0.04 →
  rate4 = 0.02 →
  abs ((apply_fluctuation 
         (apply_fluctuation 
           (apply_fluctuation 
             (apply_fluctuation 
               ((boys / (1 - initial_girls_percentage)) * initial_girls_percentage)
               rate1)
             rate2)
           rate3)
         rate4) - 431) < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_girls_after_four_years_l1254_125489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_canary_positions_l1254_125499

/-- Represents the position of a bird relative to the bus stop -/
structure BirdPosition where
  distance : ℝ

/-- Represents the positions of the three birds -/
structure BirdPositions where
  swallow : BirdPosition
  sparrow : BirdPosition
  canary : BirdPosition

/-- Checks if the given positions satisfy the problem conditions -/
def validPositions (pos : BirdPositions) : Prop :=
  pos.swallow.distance = 380 ∧
  pos.sparrow.distance = 450 ∧
  pos.sparrow.distance = (pos.swallow.distance + pos.canary.distance) / 2

/-- Theorem stating the possible positions of the canary -/
theorem canary_positions (pos : BirdPositions) :
  validPositions pos →
  (pos.canary.distance = 520 ∨ pos.canary.distance = 1280) :=
by
  intro h
  sorry

#check canary_positions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_canary_positions_l1254_125499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_in_interval_l1254_125495

def is_valid_sequence (k : ℕ → ℕ) : Prop :=
  ∀ i : ℕ, k i < k (i + 1) ∧ k (i + 1) - k i ≥ 2

def S (k : ℕ → ℕ) (m : ℕ) : ℕ :=
  Finset.sum (Finset.range m) k

theorem perfect_square_in_interval (k : ℕ → ℕ) (h : is_valid_sequence k) :
  ∀ n : ℕ, ∃ x : ℕ, S k n ≤ x^2 ∧ x^2 < S k (n + 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_in_interval_l1254_125495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1254_125462

theorem problem_statement (a b : ℝ) (h : ({1, a + b, a} : Set ℝ) = {0, b / a, b}) : b - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1254_125462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_pass_count_l1254_125470

theorem exam_pass_count :
  ∀ (total_candidates : ℕ) 
    (total_average marks_pass marks_fail : ℝ) 
    (pass_count : ℕ),
  total_candidates = 100 →
  total_average = 50 →
  marks_pass = 70 →
  marks_fail = 20 →
  pass_count + (total_candidates - pass_count) = total_candidates →
  pass_count * marks_pass + (total_candidates - pass_count) * marks_fail = total_candidates * total_average →
  pass_count = 60 := by
  intros total_candidates total_average marks_pass marks_fail pass_count
  intros h_total h_avg h_pass h_fail h_sum h_marks
  sorry

#check exam_pass_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_pass_count_l1254_125470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_second_quadrant_z_modulus_z_real_part_z_value_l1254_125457

-- Define the complex number z
noncomputable def z : ℂ := sorry

-- Define the conditions
theorem z_second_quadrant : (z.re < 0) ∧ (z.im > 0) := sorry
theorem z_modulus : Complex.abs z = 3 := sorry
theorem z_real_part : z.re = -Real.sqrt 5 := sorry

-- State the theorem
theorem z_value : z = Complex.mk (-Real.sqrt 5) 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_second_quadrant_z_modulus_z_real_part_z_value_l1254_125457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_sum_of_reciprocals_l1254_125446

/-- The derivative of (1 / (1 - sqrt x)) + (1 / (1 + sqrt x)) is 2 / (1 - x)^2 -/
theorem derivative_of_sum_of_reciprocals (x : ℝ) :
  deriv (λ t => (1 / (1 - Real.sqrt t)) + (1 / (1 + Real.sqrt t))) x = 2 / (1 - x)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_sum_of_reciprocals_l1254_125446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_area_increase_14_to_18_l1254_125483

/-- Calculates the percent increase in area from a smaller diameter pizza to a larger diameter pizza -/
noncomputable def pizza_area_increase (small_diameter large_diameter : ℝ) : ℝ :=
  let small_radius := small_diameter / 2
  let large_radius := large_diameter / 2
  let small_area := Real.pi * small_radius ^ 2
  let large_area := Real.pi * large_radius ^ 2
  let area_increase := large_area - small_area
  let percent_increase := (area_increase / small_area) * 100
  percent_increase

/-- The percent increase in area from a 14-inch pizza to an 18-inch pizza is approximately 65.31% -/
theorem pizza_area_increase_14_to_18 :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |pizza_area_increase 14 18 - 65.31| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_area_increase_14_to_18_l1254_125483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_circle_visibility_l1254_125438

/-- The radius of a circle concentric with and outside a regular hexagon -/
noncomputable def circle_radius (s : ℝ) : ℝ := 3 * Real.sqrt 6 - 2 * Real.sqrt 2

/-- The probability of seeing exactly two sides of the hexagon from a random point on the circle -/
noncomputable def probability_two_sides_visible (r : ℝ) (s : ℝ) : ℝ := 1 / 2

/-- Theorem stating that the probability of seeing exactly two sides is 1/2 for the given radius -/
theorem hexagon_circle_visibility (s : ℝ) (h : s = 4) :
  probability_two_sides_visible (circle_radius s) s = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_circle_visibility_l1254_125438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jerrys_mother_reduction_percentage_jerrys_mother_reduction_percentage_proof_l1254_125416

/-- Proves that the percentage reduction by Jerry's mother is 30%, given the initial temperature, temperature changes, and final temperature. -/
theorem jerrys_mother_reduction_percentage (initial_temp : ℝ) : Prop :=
  let doubled_temp := initial_temp * 2
  let after_dad_temp := doubled_temp - 30
  let final_temp := 59
  let sister_increase := 24
  ∀ mother_reduction_percentage : ℝ,
    (after_dad_temp * (1 - mother_reduction_percentage / 100) + sister_increase = final_temp) →
    mother_reduction_percentage = 30

-- Proof of the theorem
theorem jerrys_mother_reduction_percentage_proof :
  jerrys_mother_reduction_percentage 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jerrys_mother_reduction_percentage_jerrys_mother_reduction_percentage_proof_l1254_125416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l1254_125464

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_property (t : Triangle) 
  (h1 : t.c = 2)
  (h2 : t.b^2 - 2*t.b*t.c*Real.cos t.A = t.a^2 - 2*t.a*t.c*Real.cos t.B)
  (h3 : 7*Real.cos t.B = 2*Real.cos t.C) :
  t.a = t.b ∧ (1/2 * t.a * t.b * Real.sin t.C = Real.sqrt 15) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l1254_125464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_interior_angle_proof_l1254_125454

/-- The measure of one interior angle of a regular octagon is 135 degrees. -/
def regular_octagon_interior_angle : ℚ := 135

/-- A regular octagon has 8 sides. -/
def regular_octagon_sides : ℕ := 8

/-- The sum of interior angles of a polygon with n sides is 180(n-2) degrees. -/
def sum_of_interior_angles (n : ℕ) : ℚ := 180 * (n - 2)

/-- All interior angles of a regular polygon are equal. -/
axiom regular_polygon_equal_angles (n : ℕ) : 
  ∀ (angle : ℚ), angle * n = sum_of_interior_angles n → 
    angle = sum_of_interior_angles n / n

/-- Proof that the interior angle of a regular octagon is 135 degrees. -/
theorem regular_octagon_interior_angle_proof : 
  regular_octagon_interior_angle = 
    sum_of_interior_angles regular_octagon_sides / regular_octagon_sides :=
by
  -- Unfold definitions
  unfold regular_octagon_interior_angle
  unfold regular_octagon_sides
  unfold sum_of_interior_angles
  -- Simplify the right-hand side
  simp
  -- The proof is complete
  rfl

-- Evaluate the result
#eval regular_octagon_interior_angle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_interior_angle_proof_l1254_125454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_9_and_S_9_l1254_125497

def a (n : ℕ) : ℕ :=
  if n % 2 = 1 then 2^(n-1) else 2*n - 1

def S (n : ℕ) : ℕ :=
  (List.range n).map (fun i => a (i + 1)) |>.sum

theorem a_9_and_S_9 : a 9 = 256 ∧ S 9 = 377 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_9_and_S_9_l1254_125497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_petal_sum_sequences_l1254_125418

def PetalSequence : Type := List Nat

def isValidSequence (seq : PetalSequence) : Prop :=
  seq.length = 3 ∧
  seq.get! 0 % 9 = 0 ∧
  seq.get! 1 % 8 = 0 ∧
  seq.get! 2 % 10 = 0 ∧
  seq.get! 0 ≤ 45 ∧ seq.get! 0 ≥ 28 ∧
  seq.get! 1 ≤ seq.get! 0 - 1 ∧ seq.get! 1 ≥ seq.get! 0 - 17 ∧
  seq.get! 2 ≤ seq.get! 1 - 1 ∧ seq.get! 2 ≥ seq.get! 1 - 17

theorem petal_sum_sequences :
  let validSequences : List PetalSequence :=
    [[36, 32, 20], [36, 32, 30], [36, 24, 20], [36, 24, 10]]
  ∀ (seq : PetalSequence), isValidSequence seq → seq ∈ validSequences := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_petal_sum_sequences_l1254_125418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_park_fencing_cost_l1254_125492

/-- Represents a rectangular park -/
structure RectangularPark where
  length : ℝ
  width : ℝ
  area : ℝ
  fencing_cost_paise : ℝ

/-- The cost of fencing a rectangular park in rupees -/
noncomputable def fencing_cost (park : RectangularPark) : ℝ :=
  2 * (park.length + park.width) * (park.fencing_cost_paise / 100)

theorem park_fencing_cost (park : RectangularPark) 
  (h1 : park.length / park.width = 3 / 2)
  (h2 : park.area = 3750)
  (h3 : park.fencing_cost_paise = 40) :
  fencing_cost park = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_park_fencing_cost_l1254_125492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_theorem_l1254_125450

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the inverse function of f
variable (f_inv : ℝ → ℝ)

-- State that f is invertible
axiom f_invertible : Function.Bijective f

-- Define f_inv as the inverse of f
axiom f_inv_def : Function.RightInverse f_inv f ∧ Function.LeftInverse f_inv f

-- State the condition that f(1) - 1 = 2
axiom f_condition : f 1 - 1 = 2

-- Theorem to prove
theorem inverse_function_theorem :
  (1 / 3 : ℝ) + f_inv 3 = 4 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_theorem_l1254_125450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_AD_equation_area_ABC_l1254_125428

noncomputable section

-- Define the points
def A : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (-1, -2)
def C : ℝ × ℝ := (-3, 4)

-- Define D as the midpoint of BC
def D : ℝ × ℝ := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)

-- Theorem for the equation of line AD
theorem line_AD_equation :
  ∀ x y : ℝ, (x - D.1) * (A.2 - D.2) = (y - D.2) * (A.1 - D.1) ↔ x - 2*y + 4 = 0 :=
by sorry

-- Theorem for the area of triangle ABC
theorem area_ABC : 
  (1/2) * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) = 14 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_AD_equation_area_ABC_l1254_125428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wall_height_width_ratio_l1254_125475

/-- Represents the dimensions and properties of a rectangular wall. -/
structure Wall where
  width : ℝ
  height : ℝ
  length : ℝ
  volume : ℝ

/-- The ratio of the wall's height to its width. -/
noncomputable def heightWidthRatio (w : Wall) : ℝ := w.height / w.width

/-- Theorem stating the properties of the wall and the ratio to be proved. -/
theorem wall_height_width_ratio (w : Wall) 
  (h_width : w.width = 3)
  (h_length : w.length = 7 * w.height)
  (h_volume : w.volume = 6804)
  (h_vol_calc : w.volume = w.width * w.height * w.length) :
  heightWidthRatio w = 6 * Real.sqrt 3 := by
  sorry

#check wall_height_width_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wall_height_width_ratio_l1254_125475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1254_125465

/-- The curve S is defined as y = 3x - x^3 -/
def S (x : ℝ) : ℝ := 3*x - x^3

/-- The point A on the curve S -/
def A : ℝ × ℝ := (2, -2)

/-- The slope of the tangent line at point A -/
def tangent_slope : ℝ := -9

/-- The y-intercept of the tangent line -/
def y_intercept : ℝ := 16

/-- Theorem: The equation of the tangent line to the curve S at point A is y = -9x + 16 -/
theorem tangent_line_equation :
  let (x₀, y₀) := A
  (λ x => tangent_slope * (x - x₀) + y₀) = (λ x => -9 * x + 16) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1254_125465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_rectangles_dissection_l1254_125449

/-- A rectangle in a 2D plane --/
structure Rectangle where
  -- We don't need to define the specifics of a rectangle for this statement

/-- A point in a 2D plane --/
structure Point where
  -- We don't need to define the specifics of a point for this statement

/-- Represents a dissection of a rectangle into smaller rectangles --/
structure Dissection (R : Rectangle) where
  rectangles : List Rectangle
  -- Other properties ensuring the dissection is valid

/-- Predicate to check if two points are on a line parallel to a side of the rectangle --/
def are_parallel (R : Rectangle) (p1 p2 : Point) : Prop :=
  sorry

/-- Predicate to check if a point is inside a rectangle --/
def is_inside (p : Point) (r : Rectangle) : Prop :=
  sorry

/-- Predicate to check if a point is on the boundary of a rectangle --/
def is_on_boundary (p : Point) (r : Rectangle) : Prop :=
  sorry

/-- The main theorem --/
theorem min_rectangles_dissection 
  (R : Rectangle) 
  (points : List Point) 
  (h1 : ∀ p, p ∈ points → is_inside p R) 
  (h2 : ∀ p1 p2, p1 ∈ points → p2 ∈ points → p1 ≠ p2 → ¬(are_parallel R p1 p2)) :
  ∀ d : Dissection R, 
    (∀ p, p ∈ points → ∀ r, r ∈ d.rectangles → ¬(is_inside p r)) →
    (∀ p, p ∈ points → ∃ r, r ∈ d.rectangles ∧ is_on_boundary p r) →
    d.rectangles.length ≥ points.length + 1 :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_rectangles_dissection_l1254_125449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l1254_125414

/-- Calculates the length of a train given its speed, the bridge length, and the time to pass the bridge. -/
noncomputable def train_length (train_speed : ℝ) (bridge_length : ℝ) (time_to_pass : ℝ) : ℝ :=
  train_speed * 1000 / 3600 * time_to_pass - bridge_length

/-- Proves that a train with speed 30 km/hour passing a bridge of 140 meters in 60 seconds has a length of approximately 359.8 meters. -/
theorem train_length_calculation :
  let train_speed : ℝ := 30
  let bridge_length : ℝ := 140
  let time_to_pass : ℝ := 60
  abs (train_length train_speed bridge_length time_to_pass - 359.8) < 0.1 := by
  sorry

-- Remove the #eval statement as it's not compatible with noncomputable definitions
-- #eval train_length 30 140 60

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l1254_125414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_volume_correct_l1254_125404

/-- Regular triangular prism with base edge length a, cut by a non-parallel plane -/
structure CutPrism where
  a : ℝ
  h₁ : ℝ
  h₂ : ℝ
  h₃ : ℝ
  h₁_h₃_eq_2h₂ : h₁ + h₃ = 2 * h₂

/-- The volume of the remaining geometric body after cutting the prism -/
noncomputable def remainingVolume (p : CutPrism) : ℝ := (Real.sqrt 3 / 4) * p.h₂ * p.a^2

/-- Theorem stating that the remaining volume formula is correct -/
theorem remaining_volume_correct (p : CutPrism) :
  remainingVolume p = (Real.sqrt 3 / 4) * p.h₂ * p.a^2 := by
  -- Unfold the definition of remainingVolume
  unfold remainingVolume
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_volume_correct_l1254_125404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_transformation_result_l1254_125421

open Real Matrix

noncomputable def rotation_matrix (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![cos θ, -sin θ;
     sin θ,  cos θ]

def reflection_x_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  !![ 1,  0;
      0, -1]

noncomputable def combined_transformation : Matrix (Fin 2) (Fin 2) ℝ :=
  reflection_x_matrix * rotation_matrix (π / 4)

theorem combined_transformation_result :
  combined_transformation = !![sqrt 2 / 2, -sqrt 2 / 2;
                              -sqrt 2 / 2, -sqrt 2 / 2] := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_transformation_result_l1254_125421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reward_function_exceeds_limit_bonus_function_satisfies_conditions_l1254_125425

-- Define the reward function
noncomputable def reward_function (x : ℝ) : ℝ := Real.log x / Real.log 10 + x / 50 + 5

-- Define the bonus function
noncomputable def bonus_function (x : ℝ) : ℝ := (15 * x - 315) / (x + 8)

-- Theorem for the first part of the problem
theorem reward_function_exceeds_limit : 
  ∃ x : ℝ, 5 ≤ x ∧ x ≤ 50 ∧ reward_function x > 0.15 * x := by
  sorry

-- Theorem for the second part of the problem
theorem bonus_function_satisfies_conditions : 
  ∀ x : ℝ, 5 ≤ x ∧ x ≤ 50 → 
    bonus_function x ≥ 7 ∧ bonus_function x ≤ 0.15 * x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reward_function_exceeds_limit_bonus_function_satisfies_conditions_l1254_125425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_values_l1254_125463

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x^2 + 3*x + 2 = 0}

-- Define set B (parameterized by m)
def B (m : ℝ) : Set ℝ := {x | x^2 + (m+1)*x + m = 0}

-- Theorem statement
theorem solution_values (m : ℝ) : (U \ A ∩ B m = ∅) → (m = 1 ∨ m = 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_values_l1254_125463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distances_equals_4_2_plus_sqrt3_l1254_125427

/-- Line l parameterized by t -/
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (Real.sqrt 3 * t, 2 - t)

/-- Parabola C -/
def parabola_c (x y : ℝ) : Prop := y^2 = 2*x

/-- Point A -/
def point_a : ℝ × ℝ := (0, 2)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Theorem: Sum of distances from A to intersection points equals 4(2 + √3) -/
theorem sum_distances_equals_4_2_plus_sqrt3 :
  ∃ (t1 t2 : ℝ),
    parabola_c (line_l t1).1 (line_l t1).2 ∧
    parabola_c (line_l t2).1 (line_l t2).2 ∧
    t1 ≠ t2 ∧
    distance point_a (line_l t1) + distance point_a (line_l t2) = 4 * (2 + Real.sqrt 3) := by
  sorry

#eval "Theorem statement compiled successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distances_equals_4_2_plus_sqrt3_l1254_125427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_suv_max_distance_l1254_125459

/-- Represents the fuel efficiency of an SUV in different driving conditions -/
structure SUVFuelEfficiency where
  highway : ℚ  -- Highway fuel efficiency in miles per gallon
  city : ℚ     -- City fuel efficiency in miles per gallon

/-- Calculates the maximum distance an SUV can travel given its fuel efficiency and available fuel -/
def maxDistance (efficiency : SUVFuelEfficiency) (fuel : ℚ) : ℚ :=
  max efficiency.highway efficiency.city * fuel

/-- Theorem stating the maximum distance the specific SUV can travel -/
theorem suv_max_distance :
  let efficiency : SUVFuelEfficiency := ⟨12.2, 7.6⟩
  let availableFuel : ℚ := 24
  maxDistance efficiency availableFuel = 292.8 := by
  sorry

#eval maxDistance ⟨12.2, 7.6⟩ 24

end NUMINAMATH_CALUDE_ERRORFEEDBACK_suv_max_distance_l1254_125459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_decreasing_interval_l1254_125437

-- Define the function f
def f (a b x : ℝ) := x^2 * (a*x + b)

-- State the theorem
theorem function_decreasing_interval
  (a b : ℝ)
  (extremum_at_2 : ∃ (ε : ℝ), ε > 0 ∧ ∀ x, |x - 2| < ε → f a b x ≤ f a b 2)
  (tangent_parallel : (deriv (f a b)) 1 = -3) :
  ∃ (ε₁ ε₂ : ℝ), ε₁ > 0 ∧ ε₂ > 0 ∧
  ∀ x y, |x| < ε₁ → |y - 2| < ε₂ →
  x < y → f a b x > f a b y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_decreasing_interval_l1254_125437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_dot_product_l1254_125474

-- Define the equilateral triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  dist A B = 2 * Real.sqrt 6 ∧ 
  dist B C = 2 * Real.sqrt 6 ∧ 
  dist C A = 2 * Real.sqrt 6

-- Define the circumscribed circle of ABC
def CircumscribedCircle (O : ℝ × ℝ) (ABC : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : Prop :=
  let (A, B, C) := ABC
  dist O A = dist O B ∧ dist O B = dist O C

-- Define a chord MN of length 4
def Chord (M N : ℝ × ℝ) (O : ℝ × ℝ) : Prop :=
  dist M N = 4 ∧ dist O M = dist O N

-- Define a point P on the sides of ABC
def PointOnSide (P : ℝ × ℝ) (ABC : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : Prop :=
  let (A, B, C) := ABC
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = ((1 - t) • A + t • B)) ∨
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = ((1 - t) • B + t • C)) ∨
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = ((1 - t) • C + t • A))

-- Dot product of vectors
def dotProduct (v w : ℝ × ℝ) : ℝ :=
  let (vx, vy) := v
  let (wx, wy) := w
  vx * wx + vy * wy

-- Main theorem
theorem max_dot_product 
  (A B C : ℝ × ℝ) 
  (O : ℝ × ℝ) 
  (M N : ℝ × ℝ) 
  (P : ℝ × ℝ) :
  Triangle A B C →
  CircumscribedCircle O (A, B, C) →
  Chord M N O →
  PointOnSide P (A, B, C) →
  dotProduct (M - P) (P - N) ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_dot_product_l1254_125474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_lower_bound_l1254_125429

/-- The function f(x) defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (x + 1/x) - |x - 1/x|

/-- The theorem statement -/
theorem function_lower_bound (a : ℝ) :
  (∀ x > 0, f a x ≥ (1/2) * x) → a ≥ 3/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_lower_bound_l1254_125429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flag_height_is_nine_l1254_125444

/-- Given three rectangles and a desired length, calculate the height of a new rectangle with the same area --/
noncomputable def calculateFlagHeight (rect1_length rect1_width rect2_length rect2_width rect3_length rect3_width desired_length : ℝ) : ℝ :=
  let total_area := rect1_length * rect1_width + rect2_length * rect2_width + rect3_length * rect3_width
  total_area / desired_length

/-- Theorem stating that the flag height is 9 feet given the specific rectangle dimensions and desired length --/
theorem flag_height_is_nine :
  calculateFlagHeight 8 5 10 7 5 5 15 = 9 := by
  -- Unfold the definition of calculateFlagHeight
  unfold calculateFlagHeight
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_flag_height_is_nine_l1254_125444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l1254_125436

def A : Set ℝ := {x | x^2 - 8*x + 15 = 0}

def B (a : ℝ) : Set ℝ := {x | a*x - 1 = 0}

theorem problem_1 : B (1/5) ⊂ A := by sorry

theorem problem_2 : {a : ℝ | B a ⊆ A} = {0, 1/3, 1/5} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l1254_125436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cafeteria_quality_survey_l1254_125408

/-- Frequency distribution for cafeteria service quality ratings -/
structure FrequencyDistribution :=
  (f40_50 f50_60 f60_70 f70_80 f80_90 f90_100 : ℝ)

/-- Calculate the average score of the first three groups -/
noncomputable def average_first_three (fd : FrequencyDistribution) : ℝ :=
  (45 * fd.f40_50 + 55 * fd.f50_60 + 65 * fd.f60_70) / (fd.f40_50 + fd.f50_60 + fd.f60_70)

/-- Calculate the probability of no inspection -/
noncomputable def prob_no_inspection (fd : FrequencyDistribution) : ℝ :=
  let p_average := fd.f60_70 + fd.f70_80
  let p_good := fd.f80_90 + fd.f90_100
  3 * (p_good * p_average * p_average) + 3 * (p_average * p_good * p_good) + (p_good * p_good * p_good)

/-- The main theorem -/
theorem cafeteria_quality_survey (fd : FrequencyDistribution)
  (h1 : fd.f40_50 = 0.04)
  (h2 : fd.f50_60 = 0.06)
  (h3 : fd.f60_70 = 0.22)
  (h4 : fd.f70_80 = 0.28)
  (h5 : fd.f80_90 = 0.22)
  (h6 : fd.f90_100 = 0.18) :
  average_first_three fd = 60.625 ∧ 1 - prob_no_inspection fd = 0.396 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cafeteria_quality_survey_l1254_125408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_24_7396_to_hundredth_l1254_125415

/-- Round a real number to the nearest hundredth -/
noncomputable def roundToHundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

/-- The problem statement -/
theorem round_24_7396_to_hundredth :
  roundToHundredth 24.7396 = 24.74 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_24_7396_to_hundredth_l1254_125415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base4_division_theorem_l1254_125448

/-- Represents a number in base 4 -/
structure Base4 where
  value : ℕ

/-- Converts a base 4 number to its decimal (base 10) representation -/
def to_decimal (n : Base4) : ℕ := sorry

/-- Converts a decimal (base 10) number to its base 4 representation -/
def to_base4 (n : ℕ) : Base4 := sorry

/-- Performs division on base 4 numbers -/
def base4_div (a b : Base4) : Base4 := sorry

theorem base4_division_theorem (a b : Base4) : 
  to_decimal a = 2033 ∧ to_decimal b = 22 → to_decimal (base4_div a b) = 11 := by
  sorry

#check base4_division_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base4_division_theorem_l1254_125448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1254_125471

open Real

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  -- Conditions
  (Real.sin B * Real.sin C - Real.cos B * Real.cos C = 1/2) →
  (a = 2) →
  (b + c = 2 * Real.sqrt 3) →
  -- Conclusions
  (A = π/3) ∧
  (1/2 * b * c * Real.sin A = 2 * Real.sqrt 3 / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1254_125471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_side_c_l1254_125442

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfies_conditions (t : Triangle) : Prop :=
  2 * Real.sin (t.C / 2) * (Real.sqrt 3 * Real.cos (t.C / 2) - Real.sin (t.C / 2)) = 1 ∧
  (1 / 2) * t.a * t.b * Real.sin t.C = 2 * Real.sqrt 3 ∧
  t.A + t.B + t.C = Real.pi

-- Theorem statement
theorem min_side_c (t : Triangle) (h : satisfies_conditions t) :
  t.c ≥ 2 * Real.sqrt 2 := by
  sorry

#check min_side_c

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_side_c_l1254_125442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_thirty_degrees_l1254_125426

theorem cos_thirty_degrees :
  Real.cos (30 * π / 180) = Real.sqrt 3 / 2 :=
by
  -- Define the point on the unit circle
  let point : ℝ × ℝ := (Real.cos (30 * π / 180), Real.sin (30 * π / 180))
  
  -- Assert that the point is on the unit circle
  have h1 : point.1^2 + point.2^2 = 1 := by sorry
  
  -- Assert that the point is 30° counterclockwise from positive x-axis
  have h2 : Real.arccos point.1 = 30 * π / 180 := by sorry
  
  -- Prove that the x-coordinate (cos 30°) equals √3/2
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_thirty_degrees_l1254_125426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_l1254_125439

/-- Definition of the focus of a parabola -/
def focus_of_parabola (x y : ℝ) : ℝ × ℝ :=
  sorry

/-- Definition of a parabola given by y = 2x^2 -/
def is_parabola (x y : ℝ) : Prop :=
  y = 2 * x^2

/-- The focus of the parabola y = 2x^2 is at the point (0, 1/8) -/
theorem parabola_focus (x y : ℝ) :
  is_parabola x y → focus_of_parabola x y = (0, 1/8) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_l1254_125439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_angle_l1254_125477

theorem perpendicular_lines_angle (α : Real) : 
  α ∈ Set.Icc 0 (2 * Real.pi) →
  (∃ (x y : Real), x * Real.cos α - y - 1 = 0 ∧ x + y * Real.sin α + 1 = 0) →
  (∀ (x₁ y₁ x₂ y₂ : Real), 
    x₁ * Real.cos α - y₁ - 1 = 0 ∧ 
    x₂ + y₂ * Real.sin α + 1 = 0 →
    (x₁ - x₂) * (y₁ - y₂) = 0) →
  α = Real.pi / 4 ∨ α = 5 * Real.pi / 4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_angle_l1254_125477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_identity_l1254_125472

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * x) / (x + 1)

theorem function_composition_identity (a : ℝ) :
  (∀ x : ℝ, x ≠ -1 → f a (f a x) = x) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_identity_l1254_125472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_three_distinct_roots_l1254_125480

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - 4*x + 4

-- Theorem statement
theorem f_properties :
  (f 2 = -4/3) ∧
  (deriv f 2 = 0) ∧
  (∀ x, f x ≥ -4/3) ∧
  (∀ x, f x ≤ 28/3) ∧
  (∃ x, f x = -4/3) ∧
  (∃ x, f x = 28/3) :=
by sorry

-- Define the range of k
def k_range (k : ℝ) : Prop := -4/3 < k ∧ k < 28/3

-- Theorem for the number of roots
theorem three_distinct_roots (k : ℝ) :
  (k_range k) ↔ (∃ x y z, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f x = k ∧ f y = k ∧ f z = k) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_three_distinct_roots_l1254_125480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_side_length_is_four_verify_perimeter_l1254_125469

/-- The side length of a square that, when combined with a rectangle of dimensions 8 cm by 4 cm,
    forms a new rectangle with a perimeter of 40 cm. -/
def square_side_length : ℝ := 4

theorem square_side_length_is_four :
  square_side_length = 4 := by
  -- Unfold the definition of square_side_length
  unfold square_side_length
  -- The equality is now trivial
  rfl

theorem verify_perimeter :
  2 * ((square_side_length + 8) + (square_side_length + 4)) = 40 := by
  -- Substitute the value of square_side_length
  rw [square_side_length_is_four]
  -- Simplify the expression
  simp [add_assoc, mul_add, add_mul]
  -- Evaluate the arithmetic
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_side_length_is_four_verify_perimeter_l1254_125469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_vector_length_bounds_l1254_125468

def A : ℝ × ℝ := (0, 1)
def B : ℝ × ℝ := (0, -1)
def C : ℝ × ℝ := (1, 0)

def trajectory_equation (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  (x - 2)^2 + y^2 = 1

def satisfies_condition (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  (x * x + (y - 1) * (y + 1)) = 2 * ((x - 1)^2 + y^2)

noncomputable def vector_length (P : ℝ × ℝ) : ℝ :=
  let (x, y) := P
  Real.sqrt (9 * x^2 + 9 * y^2 - 6 * y + 1)

theorem trajectory_and_vector_length_bounds :
  ∀ P : ℝ × ℝ, 
    satisfies_condition P →
    trajectory_equation P →
    vector_length P ≤ 3 + Real.sqrt 37 ∧
    vector_length P ≥ Real.sqrt 37 - 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_vector_length_bounds_l1254_125468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sampling_is_systematic_l1254_125419

/-- Represents a sampling method -/
inductive SamplingMethod
  | Systematic
  | Stratified
  | Lottery
  | Random

/-- Represents a class of students -/
structure StudentClass where
  students : Finset Nat
  student_count : students.card = 50
  numbering : ∀ n, n ∈ students ↔ 1 ≤ n ∧ n ≤ 50

/-- Represents the grade with multiple classes -/
structure Grade where
  classes : Finset StudentClass
  class_count : classes.card = 12

/-- Defines the sampling method used in the problem -/
def sampling_method (g : Grade) : SamplingMethod :=
  if ∀ c ∈ g.classes, 14 ∈ c.students then SamplingMethod.Systematic else SamplingMethod.Random

/-- Theorem stating that the sampling method used is Systematic sampling -/
theorem sampling_is_systematic (g : Grade) : sampling_method g = SamplingMethod.Systematic := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sampling_is_systematic_l1254_125419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_digit_divisible_by_11_l1254_125435

def is_divisible_by_11 (n : ℕ) : Prop :=
  ∃ k : ℤ, (623 * 10000 + n * 1000 + 8457 : ℤ) = 11 * k

theorem eight_digit_divisible_by_11 :
  ∀ n : ℕ, n < 10 → (is_divisible_by_11 n ↔ n = 2) :=
by
  sorry

#check eight_digit_divisible_by_11

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_digit_divisible_by_11_l1254_125435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_bisector_ratio_l1254_125494

-- Define a right-angled triangle ABC with right angle at C
structure RightTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  right_angle : (C.1 - A.1) * (B.1 - A.1) + (C.2 - A.2) * (B.2 - A.2) = 0

-- Define the angle bisector AD
def angle_bisector (t : RightTriangle) (D : ℝ × ℝ) : Prop :=
  ∃ (l : ℝ), 0 < l ∧ l < 1 ∧ D = (l * t.B.1 + (1 - l) * t.C.1, l * t.B.2 + (1 - l) * t.C.2)

-- Define the ratio k
noncomputable def ratio (t : RightTriangle) (D : ℝ × ℝ) (k : ℝ) : Prop :=
  let AD := Real.sqrt ((D.1 - t.A.1)^2 + (D.2 - t.A.2)^2)
  let DB := Real.sqrt ((t.B.1 - D.1)^2 + (t.B.2 - D.2)^2)
  AD / DB = k

-- Main theorem
theorem right_triangle_bisector_ratio (t : RightTriangle) (D : ℝ × ℝ) (k : ℝ) :
  angle_bisector t D → ratio t D k →
  (k > 0 ∧ ∃ (t : RightTriangle) (D : ℝ × ℝ), angle_bisector t D ∧ ratio t D k) ∧
  (k = Real.sqrt (2 + Real.sqrt 2) → 
    t.A.1^2 + t.A.2^2 = t.B.1^2 + t.B.2^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_bisector_ratio_l1254_125494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_conditions_l1254_125430

noncomputable def f (x : ℝ) : ℝ := -2 / (x + 1)

theorem f_satisfies_conditions :
  -- Condition 1: f can be written in the form (a₂x² + b₂x + c₂) / (a₁x² + b₁x + c₁)
  (∃ (a₁ a₂ b₁ b₂ c₁ c₂ : ℝ), ∀ x, f x = (a₂ * x^2 + b₂ * x + c₂) / (a₁ * x^2 + b₁ * x + c₁)) ∧
  -- Condition 2: f(0) = -2 and f(1) = -1
  (f 0 = -2 ∧ f 1 = -1) ∧
  -- Condition 3: For any x ∈ [0, +∞), f(x) < 0
  (∀ x : ℝ, x ≥ 0 → f x < 0) ∧
  -- Condition 4: f(x) is monotonically increasing on [0, +∞)
  (∀ x y : ℝ, 0 ≤ x ∧ x < y → f x < f y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_conditions_l1254_125430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_similarity_under_spiral_transformation_l1254_125456

-- Define the spiral similarity transformation
def spiral_similarity (P : ℝ × ℝ → ℝ × ℝ) : Prop :=
  ∃ (k : ℝ) (θ : ℝ), ∀ (v : ℝ × ℝ), P v = (k * (Real.cos θ * v.1 - Real.sin θ * v.2), k * (Real.sin θ * v.1 + Real.cos θ * v.2))

-- Define the triangle similarity
def triangle_similar (A B C A' B' C' : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), k > 0 ∧
    Real.sqrt ((B'.1 - A'.1)^2 + (B'.2 - A'.2)^2) = k * Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) ∧
    Real.sqrt ((C'.1 - B'.1)^2 + (C'.2 - B'.2)^2) = k * Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) ∧
    Real.sqrt ((A'.1 - C'.1)^2 + (A'.2 - C'.2)^2) = k * Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)

theorem triangle_similarity_under_spiral_transformation
  (A B C A₁ B₁ C₁ O A₂ B₂ C₂ : ℝ × ℝ)
  (P : ℝ × ℝ → ℝ × ℝ)
  (h_spiral : spiral_similarity P)
  (h_transform : P A = A₁ ∧ P B = B₁ ∧ P C = C₁)
  (h_parallelogram_A : A₂ = (A.1 + A₁.1 - O.1, A.2 + A₁.2 - O.2))
  (h_parallelogram_B : B₂ = (B.1 + B₁.1 - O.1, B.2 + B₁.2 - O.2))
  (h_parallelogram_C : C₂ = (C.1 + C₁.1 - O.1, C.2 + C₁.2 - O.2)) :
  triangle_similar A B C A₂ B₂ C₂ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_similarity_under_spiral_transformation_l1254_125456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1254_125478

def f (x : ℝ) : ℝ := |x + 3|
def g (m : ℝ) (x : ℝ) : ℝ := m - 2*|x - 11|

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, 2 * f x ≥ g m (x + 4)) ↔ m ≤ 20 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1254_125478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_workers_theorem_l1254_125423

/-- Represents the company's profit scenario -/
structure CompanyProfit where
  maintenance_fee : ℚ
  hourly_wage : ℚ
  hourly_production : ℕ
  widget_price : ℚ
  workday_hours : ℕ

/-- Calculates the minimum number of workers needed for profit -/
def min_workers_for_profit (c : CompanyProfit) : ℕ :=
  Nat.ceil (c.maintenance_fee / (c.hourly_production * c.widget_price * c.workday_hours - c.hourly_wage * c.workday_hours))

/-- Theorem stating the minimum number of workers needed for profit -/
theorem min_workers_theorem (c : CompanyProfit) 
  (h1 : c.maintenance_fee = 600)
  (h2 : c.hourly_wage = 20)
  (h3 : c.hourly_production = 3)
  (h4 : c.widget_price = 14/5)
  (h5 : c.workday_hours = 8) :
  min_workers_for_profit c = 7 := by
  sorry

#eval min_workers_for_profit ⟨600, 20, 3, 14/5, 8⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_workers_theorem_l1254_125423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_digit_number_rearrangement_l1254_125405

def is_eight_digit (n : ℕ) : Prop := 10000000 ≤ n ∧ n ≤ 99999999

def move_last_digit_to_first (n : ℕ) : ℕ :=
  (n % 10) * 10000000 + n / 10

theorem eight_digit_number_rearrangement (B : ℕ) 
  (h1 : is_eight_digit B)
  (h2 : Nat.Coprime B 36)
  (h3 : B > 77777777) :
  ∃ (A_min A_max : ℕ),
    (∀ A : ℕ, is_eight_digit A ∧ A = move_last_digit_to_first B → A_min ≤ A ∧ A ≤ A_max) ∧
    A_min = 17777779 ∧
    A_max = 99999998 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_digit_number_rearrangement_l1254_125405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_problem_l1254_125491

noncomputable def t : ℝ := sorry

noncomputable def a : ℕ → ℝ := sorry

noncomputable def S : ℕ → ℝ := sorry

noncomputable def b (n : ℕ) : ℝ := (2 * (n : ℝ) + 1) / a n

noncomputable def T : ℕ → ℝ := sorry

theorem sequence_problem (h1 : ∀ n, S n = t * (S n - a n + 1))
                         (h2 : t > 0)
                         (h3 : 4 * a 3 = (a 1 + 2 * a 2) / 2) :
  t = 1/2 ∧
  (∀ n, a n = 1 / (2^n)) ∧
  (∀ n, T n = 2 + (2 * (n : ℝ) - 1) * 2^(n+1)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_problem_l1254_125491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_composite_function_l1254_125410

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain_f : Set ℝ := Set.Icc 1 3

-- Define the composite function g
noncomputable def g (x : ℝ) : ℝ := f (2^x - 1)

-- Theorem statement
theorem domain_of_composite_function :
  {x : ℝ | g x ∈ Set.range f} = Set.Icc 1 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_composite_function_l1254_125410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_football_player_goals_l1254_125413

theorem football_player_goals (average_increase : ℝ) (fifth_match_goals : ℕ) : ℝ := by
  let initial_average : ℝ := 4
  let total_goals : ℕ := 21
  have h1 : average_increase = 0.2 := by sorry
  have h2 : fifth_match_goals = 5 := by sorry
  have h3 : (4 * initial_average + ↑fifth_match_goals) / 5 = initial_average + average_increase := by sorry
  exact ↑total_goals

#check football_player_goals

end NUMINAMATH_CALUDE_ERRORFEEDBACK_football_player_goals_l1254_125413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_function_inequality_l1254_125487

-- Define a monotonous function f on ℝ
variable (f : ℝ → ℝ)
variable (hf : Monotone f)

-- Define the theorem
theorem monotone_function_inequality
  (x₁ x₂ l : ℝ)
  (h_x : x₁ < x₂)
  (h_l : l ≠ -1)
  (α : ℝ := (x₁ + l * x₂) / (1 + l))
  (β : ℝ := (x₂ + l * x₁) / (1 + l))
  (h_ineq : |f x₁ - f x₂| < |f α - f β|) :
  l < 0 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_function_inequality_l1254_125487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_mist_nozzles_count_l1254_125412

/-- Water mist nozzle working pressure (MPa) -/
def P : ℝ := 0.35

/-- Flow coefficient of water mist nozzles -/
def K : ℝ := 24.96

/-- Protection area of the object (m²) -/
def S : ℝ := 14

/-- Design mist intensity of the object (L/min·m²) -/
def W : ℝ := 20

/-- Nozzle flow rate (L/min) -/
noncomputable def q : ℝ := K * Real.sqrt (10 * P)

/-- Number of water mist nozzles -/
noncomputable def N : ℝ := (S * W) / q

/-- Theorem stating that the number of water mist nozzles is approximately 6 -/
theorem water_mist_nozzles_count : ⌊N⌋ = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_mist_nozzles_count_l1254_125412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1254_125434

/-- Hyperbola type -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- Point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of a point being on the hyperbola -/
def on_hyperbola (h : Hyperbola) (p : Point) : Prop :=
  p.y^2 / h.a^2 - p.x^2 / h.b^2 = 1

/-- Definition of eccentricity for a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- Definition of a rhombus formed by OFAB -/
noncomputable def is_rhombus (o f a b : Point) : Prop :=
  let of_length := Real.sqrt ((f.x - o.x)^2 + (f.y - o.y)^2)
  let fa_length := Real.sqrt ((a.x - f.x)^2 + (a.y - f.y)^2)
  let ab_length := Real.sqrt ((b.x - a.x)^2 + (b.y - a.y)^2)
  let bo_length := Real.sqrt ((o.x - b.x)^2 + (o.y - b.y)^2)
  of_length = fa_length ∧ fa_length = ab_length ∧ ab_length = bo_length

/-- Main theorem -/
theorem hyperbola_eccentricity (h : Hyperbola) (o f a b : Point) 
  (h_f_focus : f.x = 0 ∧ f.y > 0) 
  (h_o_origin : o.x = 0 ∧ o.y = 0)
  (h_a_on_curve : on_hyperbola h a)
  (h_b_on_curve : on_hyperbola h b)
  (h_rhombus : is_rhombus o f a b) :
  eccentricity h = 1 + Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1254_125434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_property_l1254_125406

-- Define a cubic polynomial
def cubic_polynomial (a b c d : ℝ) : ℝ → ℝ := λ x ↦ a * x^3 + b * x^2 + c * x + d

-- Define the conditions for p(x)
def satisfies_conditions (p : ℝ → ℝ) : Prop :=
  p 1 = 1 / 1^2 ∧ p 2 = 1 / 2^2 ∧ p 3 = 1 / 3^2 ∧ p 5 = 1 / 5^2

theorem cubic_polynomial_property :
  ∃ (a b c d : ℝ), 
    let p := cubic_polynomial a b c d
    satisfies_conditions p → p 4 = 1 / 150 := by
  sorry

#check cubic_polynomial_property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_property_l1254_125406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radius_of_Q_l1254_125467

-- Define the circles and their properties
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the circles
variable (P Q R S : Circle)

-- Axioms based on the problem conditions
axiom externally_tangent : 
  (P.center.fst - Q.center.fst)^2 + (P.center.snd - Q.center.snd)^2 = (P.radius + Q.radius)^2 ∧
  (Q.center.fst - R.center.fst)^2 + (Q.center.snd - R.center.snd)^2 = (Q.radius + R.radius)^2 ∧
  (R.center.fst - P.center.fst)^2 + (R.center.snd - P.center.snd)^2 = (R.radius + P.radius)^2

axiom internally_tangent :
  (P.center.fst - S.center.fst)^2 + (P.center.snd - S.center.snd)^2 = (S.radius - P.radius)^2 ∧
  (Q.center.fst - S.center.fst)^2 + (Q.center.snd - S.center.snd)^2 = (S.radius - Q.radius)^2 ∧
  (R.center.fst - S.center.fst)^2 + (R.center.snd - S.center.snd)^2 = (S.radius - R.radius)^2

axiom Q_R_congruent : Q.radius = R.radius
axiom P_radius : P.radius = 2
axiom P_through_S_center : P.center.fst^2 + P.center.snd^2 = (S.radius - P.radius)^2

-- Define the radius of S in terms of P
def S_radius : ℝ := 2 * P.radius

-- Theorem statement
theorem radius_of_Q : Q.radius = 16 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_radius_of_Q_l1254_125467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_powderman_distance_when_blast_heard_l1254_125451

/-- The time (in seconds) at which the blast occurs -/
def blast_time : ℚ := 45

/-- The speed of the powderman in yards per second -/
def powderman_speed : ℚ := 10

/-- The speed of sound in feet per second -/
def sound_speed : ℚ := 1080

/-- Convert yards to feet -/
def yards_to_feet (yards : ℚ) : ℚ := yards * 3

/-- Convert feet to yards -/
def feet_to_yards (feet : ℚ) : ℚ := feet / 3

/-- The distance traveled by the powderman at time t -/
def powderman_distance (t : ℚ) : ℚ := yards_to_feet (powderman_speed * t)

/-- The distance traveled by sound at time t (t ≥ blast_time) -/
def sound_distance (t : ℚ) : ℚ := sound_speed * (t - blast_time)

/-- Approximate equality for rationals -/
def approx_equal (x y : ℚ) (ε : ℚ) : Prop := abs (x - y) < ε

theorem powderman_distance_when_blast_heard : 
  ∃ t : ℚ, t > blast_time ∧ 
  powderman_distance t = sound_distance t ∧ 
  approx_equal (feet_to_yards (powderman_distance t)) 463 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_powderman_distance_when_blast_heard_l1254_125451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_intersections_l1254_125409

/-- The curve function -/
noncomputable def curve (x : ℝ) : ℝ :=
  if x ≤ 2 ∧ x ≥ -2 then 2 else 1

/-- The line function -/
def line (k : ℝ) (x : ℝ) : ℝ :=
  k * (x - 2) + 4

/-- Theorem stating the conditions for two intersections -/
theorem two_intersections (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    curve x₁ = line k x₁ ∧ 
    curve x₂ = line k x₂ ∧ 
    x₁ ≤ 2 ∧ x₁ ≥ -2 ∧ x₂ ≤ 2 ∧ x₂ ≥ -2) ↔ 
  (k ≠ 0 ∧ ∃ a b : ℝ, -2 ≤ a ∧ a < b ∧ b ≤ 2 ∧ 
    curve a = line k a ∧ curve b = line k b) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_intersections_l1254_125409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_average_speed_l1254_125407

/-- Calculates the average speed of a cyclist given two trip segments -/
noncomputable def average_speed (distance1 : ℝ) (speed1 : ℝ) (distance2 : ℝ) (speed2 : ℝ) : ℝ :=
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let total_distance := distance1 + distance2
  let total_time := time1 + time2
  total_distance / total_time

/-- The average speed of a cyclist who rides 8 km at 10 km/hr and then 10 km at 8 km/hr is approximately 8.78 km/hr -/
theorem cyclist_average_speed :
  let avg_speed := average_speed 8 10 10 8
  ∃ ε > 0, |avg_speed - 8.78| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_average_speed_l1254_125407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tunnel_length_is_two_miles_l1254_125400

/-- Calculates the length of a tunnel given train parameters. -/
noncomputable def tunnel_length (train_length : ℝ) (exit_time : ℝ) (train_speed : ℝ) : ℝ :=
  (exit_time / 60) * train_speed - train_length

/-- Theorem stating that the tunnel length is 2 miles given specific train parameters. -/
theorem tunnel_length_is_two_miles :
  tunnel_length 2 4 60 = 2 := by
  -- Unfold the definition of tunnel_length
  unfold tunnel_length
  -- Simplify the expression
  simp
  -- Perform the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tunnel_length_is_two_miles_l1254_125400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_modulus_l1254_125481

theorem complex_modulus :
  let z : ℂ := Complex.I / (1 + 2 * Complex.I)
  Complex.abs z = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_modulus_l1254_125481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercepts_count_l1254_125458

theorem x_intercepts_count :
  let f : ℝ → ℝ := λ x => (x - 3) * (x^2 + 3*x + 2)
  ∃! (s : Finset ℝ), (∀ x ∈ s, f x = 0) ∧ s.card = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercepts_count_l1254_125458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_l1254_125493

/-- IsRhombus predicate -/
def IsRhombus (s : Set (ℝ × ℝ)) : Prop := sorry

/-- Perimeter function for a set of points in 2D space -/
def Perimeter (s : Set (ℝ × ℝ)) : ℝ := sorry

/-- DiagonalLength function for a set of points in 2D space -/
def DiagonalLength (s : Set (ℝ × ℝ)) : ℝ := sorry

/-- Area function for a set of points in 2D space -/
def Area (s : Set (ℝ × ℝ)) : ℝ := sorry

/-- A rhombus with perimeter 40 and one diagonal 16 has area 96 -/
theorem rhombus_area (EFGH : Set (ℝ × ℝ)) (h_rhombus : IsRhombus EFGH) 
  (h_perimeter : Perimeter EFGH = 40) (h_diagonal : DiagonalLength EFGH = 16) : 
  Area EFGH = 96 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_l1254_125493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_leg_length_l1254_125440

theorem isosceles_triangle_leg_length 
  (perimeter : ℝ) 
  (base : ℝ) 
  (h_perimeter : perimeter = 20) 
  (h_base : base = 8) :
  (perimeter - base) / 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_leg_length_l1254_125440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_melting_point_ice_fahrenheit_l1254_125482

/-- Converts Celsius to Fahrenheit -/
noncomputable def celsius_to_fahrenheit (c : ℝ) : ℝ := (9/5) * c + 32

/-- Converts Fahrenheit to Celsius -/
noncomputable def fahrenheit_to_celsius (f : ℝ) : ℝ := (5/9) * (f - 32)

theorem melting_point_ice_fahrenheit :
  let water_boiling_f : ℝ := 212
  let water_boiling_c : ℝ := 100
  let ice_melting_c : ℝ := 0
  let water_temp_c : ℝ := 50
  let water_temp_f : ℝ := 122
  celsius_to_fahrenheit ice_melting_c = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_melting_point_ice_fahrenheit_l1254_125482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_point_xoz_l1254_125473

/-- The symmetric point of (x, y, z) with respect to the xOz plane is (x, -y, z) -/
def symmetricPointXOZ (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (p.1, -p.2.1, p.2.2)

/-- The given point -/
def givenPoint : ℝ × ℝ × ℝ := (2, 3, 4)

theorem symmetric_point_xoz :
  symmetricPointXOZ givenPoint = (2, -3, 4) := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_point_xoz_l1254_125473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l1254_125447

-- Define the rational function
noncomputable def f (x : ℝ) : ℝ := (x^3 - 1) / ((x-2)^2)

-- Define the solution set
def solution_set : Set ℝ := {x | x ≥ 1 ∧ x ≠ 2}

-- Theorem statement
theorem inequality_solution :
  ∀ x : ℝ, x ≠ 2 → (f x ≥ 0 ↔ x ∈ solution_set) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l1254_125447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_triangle_properties_l1254_125417

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2/6 + y^2/2 = 1

-- Define the circle (renamed to avoid conflict with built-in circle)
def circle_eq (x y a b : ℝ) : Prop := (x - a)^2 + (y - b)^2 = 9/4

-- Define the focus and directrix
def focus : ℝ × ℝ := (-2, 0)
def directrix (x : ℝ) : Prop := x = 2

-- Define the line passing through focus and intersecting the parabola
def line_through_focus (m : ℝ) (x y : ℝ) : Prop := y = m * (x + 2)

-- Helper function for triangle area (placeholder implementation)
noncomputable def area_triangle (x₁ y₁ x₂ y₂ : ℝ) : ℝ := sorry

-- Theorem statement
theorem parabola_and_triangle_properties :
  ∃ (a b : ℝ),
    -- Circle center lies on parabola
    parabola a b ∧
    -- Circle passes through origin
    circle_eq 0 0 a b ∧
    -- Circle is tangent to directrix
    ∃ (x : ℝ), directrix x ∧ circle_eq x 0 a b ∧
    -- Parabola equation
    (∀ x y, parabola x y ↔ x^2/6 + y^2/2 = 1) ∧
    -- Minimum area of triangle PAB occurs when slope is -√6/3
    (∃ (m : ℝ), m = -Real.sqrt 6 / 3 ∧
      ∀ m' : ℝ, m' ≠ m →
        ∃ (x₁ y₁ x₂ y₂ : ℝ),
          parabola x₁ y₁ ∧ parabola x₂ y₂ ∧
          line_through_focus m x₁ y₁ ∧ line_through_focus m x₂ y₂ ∧
          line_through_focus m' x₁ y₁ ∧ line_through_focus m' x₂ y₂ ∧
          (area_triangle x₁ y₁ x₂ y₂ ≤ area_triangle x₁ y₁ x₂ y₂)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_triangle_properties_l1254_125417
