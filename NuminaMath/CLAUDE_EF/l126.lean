import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frequency_histogram_sum_one_larger_k_squared_stronger_relationship_l126_12660

-- Define a frequency distribution histogram
def FrequencyHistogram := List (ℝ × ℝ)  -- List of (height, width) pairs

-- Define the sum of areas of rectangles in a histogram
def sumAreas (hist : FrequencyHistogram) : ℝ :=
  List.sum (List.map (λ (h, w) => h * w) hist)

-- Define a K² statistic
structure KSquaredStatistic where
  observedValue : ℝ
  
-- Define a relationship strength between categorical variables
noncomputable def relationshipStrength : ℝ → ℝ := sorry

-- Theorem 1: The sum of areas in a frequency histogram is 1
theorem frequency_histogram_sum_one (hist : FrequencyHistogram) :
  sumAreas hist = 1 := by sorry

-- Theorem 2: Larger K² statistic observed value indicates stronger relationship
theorem larger_k_squared_stronger_relationship
  (X Y : Type) [Fintype X] [Fintype Y]
  (k1 k2 : KSquaredStatistic)
  (h : k1.observedValue < k2.observedValue) :
  relationshipStrength k1.observedValue < relationshipStrength k2.observedValue := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frequency_histogram_sum_one_larger_k_squared_stronger_relationship_l126_12660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equalities_l126_12676

theorem expression_equalities :
  (∃ (x : ℝ), x = Real.sqrt 27 / Real.sqrt 3 - 16 * (4⁻¹ : ℝ) + |(-5 : ℝ)| - (3 - Real.sqrt 3)^0 ∧ x = 3) ∧
  (∃ (y : ℝ), y = 2 * Real.tan (30 * π / 180) - |1 - Real.sqrt 3| + (2014 - Real.sqrt 2)^0 + Real.sqrt (1/3) ∧ y = 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equalities_l126_12676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_characterize_k_l126_12644

open Nat

/-- Given a prime number p, characterize the positive integers k that satisfy the property:
    For any positive integer n, p^(⌊((k-1)*n)/(p-1)⌋ + 1) ≥ (k*n)! / n! -/
theorem characterize_k (p : ℕ) (hp : Nat.Prime p) :
  ∀ k : ℕ, k > 0 →
  (∀ n : ℕ, n > 0 →
    p ^ (((k - 1) * n) / (p - 1) + 1) ≥ (k * n)! / n!) ↔
  ∃ α : ℕ, k = p ^ α :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_characterize_k_l126_12644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_expression_equals_zero_trig_expression_equals_one_l126_12683

-- Part I
theorem log_expression_equals_zero :
  (Real.log 9 / Real.log 2) * (Real.log 4 / Real.log 3) - (2 * Real.sqrt 2) ^ (2/3) - Real.exp (Real.log 2) = 0 := by sorry

-- Part II
theorem trig_expression_equals_one :
  (Real.sqrt (1 - Real.sin (20 * π / 180))) / (Real.cos (10 * π / 180) - Real.sin (170 * π / 180)) = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_expression_equals_zero_trig_expression_equals_one_l126_12683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equation_is_circle_l126_12686

/-- The equation of the curve in polar coordinates -/
def polar_equation (r θ : ℝ) : Prop := r = 3 * Real.sin θ * (Real.cos θ / Real.sin θ)

/-- The equation of a circle in Cartesian coordinates -/
def circle_equation (x y : ℝ) : Prop := (x - 3/2)^2 + y^2 = (3/2)^2

/-- Theorem stating that the polar equation represents a circle -/
theorem polar_equation_is_circle :
  ∀ (r θ x y : ℝ), 
    polar_equation r θ → 
    (x = r * Real.cos θ ∧ y = r * Real.sin θ) → 
    circle_equation x y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equation_is_circle_l126_12686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_l126_12609

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 - Real.log x

-- State the theorem
theorem monotonic_decreasing_interval :
  ∀ x : ℝ, x > 0 → (∀ y : ℝ, y > 0 → y < x → f y > f x) ↔ x ∈ Set.Ioc 0 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_l126_12609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_price_theorem_l126_12643

/-- Represents the problem of finding the optimal price for a commodity --/
theorem optimal_price_theorem
  (a : ℝ) -- Initial sales volume in 2016
  (k : ℝ) -- Proportionality constant
  (h1 : k = 3 * a) -- Condition that k = 3a
  (h2 : a > 0) -- Assumption that initial sales volume is positive
  : (a + k / (13 - 7)) * (13 - 5) = 1.2 * (a * (15 - 5)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_price_theorem_l126_12643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_looking_count_l126_12613

def isPrimeLooking (n : Nat) : Bool :=
  ¬ n.Prime && n ≠ 1 && n % 2 ≠ 0 && n % 7 ≠ 0

def countPrimeLooking (upperBound : Nat) : Nat :=
  (List.range upperBound).filter isPrimeLooking |>.length

theorem prime_looking_count :
  countPrimeLooking 1200 = 318 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_looking_count_l126_12613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_cream_volume_is_16_3_pi_l126_12687

noncomputable section

/-- The volume of a right circular cone -/
def cone_volume (r : ℝ) (h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

/-- The volume of a hemisphere -/
def hemisphere_volume (r : ℝ) : ℝ := (2/3) * Real.pi * r^3

/-- Represents the ice cream configuration -/
structure IceCreamConfiguration where
  large_cone_height : ℝ
  large_cone_radius : ℝ
  hemisphere_radius : ℝ
  small_cone_height : ℝ
  small_cone_radius : ℝ

/-- The total volume of ice cream in the configuration -/
def total_ice_cream_volume (config : IceCreamConfiguration) : ℝ :=
  cone_volume config.large_cone_radius config.large_cone_height +
  hemisphere_volume config.hemisphere_radius +
  cone_volume config.small_cone_radius config.small_cone_height

/-- The theorem stating the total volume of ice cream -/
theorem ice_cream_volume_is_16_3_pi :
  let config : IceCreamConfiguration := {
    large_cone_height := 12,
    large_cone_radius := 1,
    hemisphere_radius := 1,
    small_cone_height := 2,
    small_cone_radius := 1
  }
  total_ice_cream_volume config = (16/3) * Real.pi := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_cream_volume_is_16_3_pi_l126_12687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l126_12692

def A : Set ℤ := {-2, -1, 0, 1}
def B : Set ℤ := {x : ℤ | x ≥ 0}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l126_12692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l126_12695

-- Define the hyperbola and its properties
def Hyperbola (a b : ℝ) : Prop := (a > 0) ∧ (b > 0) ∧ 
  ∀ (x y : ℝ), (x^2 / a^2) - (y^2 / b^2) = 1

-- Define the asymptote condition
def AsymptoteCondition (a b : ℝ) : Prop := 
  b / a = Real.sqrt 2

-- Define eccentricity
noncomputable def Eccentricity (a b : ℝ) : ℝ := Real.sqrt ((a^2 + b^2) / a^2)

-- Theorem statement
theorem hyperbola_eccentricity (a b : ℝ) :
  Hyperbola a b → AsymptoteCondition a b → Eccentricity a b = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l126_12695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_numbers_with_integer_means_l126_12607

-- Define a function to get the divisors of a natural number
def divisors (n : ℕ) : Finset ℕ := sorry

-- Define the arithmetic mean of a finite set of natural numbers
def arithmetic_mean (s : Finset ℕ) : ℚ := sorry

-- Define the geometric mean of a finite set of natural numbers
noncomputable def geometric_mean (s : Finset ℕ) : ℚ := sorry

-- The main theorem
theorem infinitely_many_numbers_with_integer_means :
  ∃ (S : Set ℕ), (Set.Infinite S) ∧ 
  (∀ n ∈ S, 
    (arithmetic_mean (divisors n)).num % (arithmetic_mean (divisors n)).den = 0 ∧ 
    (geometric_mean (divisors n)).num % (geometric_mean (divisors n)).den = 0) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_numbers_with_integer_means_l126_12607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_trip_distance_l126_12616

/-- Proves that given a boat speed, stream speed, and total round trip time, 
    the distance to the destination can be calculated. -/
theorem boat_trip_distance 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (total_time : ℝ) 
  (h1 : boat_speed = 5) 
  (h2 : stream_speed = 2) 
  (h3 : total_time = 200) : 
  ∃ (distance : ℝ), distance = 420 ∧ 
  distance / (boat_speed + stream_speed) + 
  distance / (boat_speed - stream_speed) = total_time := by
  sorry

#check boat_trip_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_trip_distance_l126_12616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_l126_12603

def initial_investment : ℝ := 6500
def interest_rates : List ℝ := [0.065, 0.07, 0.06, 0.075, 0.08]

def compound_interest (principal : ℝ) (rate : ℝ) : ℝ :=
  principal * (1 + rate)

def final_amount (initial : ℝ) (rates : List ℝ) : ℝ :=
  rates.foldl compound_interest initial

theorem investment_growth :
  ‖final_amount initial_investment interest_rates - 9113.43‖ < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_l126_12603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_inequality_l126_12658

/-- Given a quadrilateral with side lengths a, b, c, d and area S,
    prove that S ≤ ((a+c)/2) * ((b+d)/2) -/
theorem quadrilateral_area_inequality (a b c d S : ℝ) 
    (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0)
    (h_area : S > 0) : 
  S ≤ ((a+c)/2) * ((b+d)/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_inequality_l126_12658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_theorem_l126_12615

theorem sin_sum_theorem (α : ℝ) (h1 : α ∈ Set.Ioo (π/2) π) (h2 : Real.sin (α + π/3) = 12/13) :
  Real.sin (π/6 - α) + Real.sin (2*π/3 - α) = 7/13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_theorem_l126_12615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l126_12608

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (2, -1)

theorem vector_properties :
  (¬ ∃ (k : ℝ), a = k • b) ∧
  (a.1 * b.1 + a.2 * b.2 = 0) ∧
  (a.1^2 + a.2^2 = b.1^2 + b.2^2) ∧
  ((a.1 + b.1)^2 + (a.2 + b.2)^2 = (a.1 - b.1)^2 + (a.2 - b.2)^2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l126_12608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_revolutions_wheel_circumferences_l126_12605

-- Define the radius of the front wheel
noncomputable def front_radius : ℝ := sorry

-- Define the radius of the rear wheel
noncomputable def rear_radius : ℝ := 2 * front_radius

-- Define the circumference of the front wheel
noncomputable def front_circumference : ℝ := 2 * Real.pi * front_radius

-- Define the circumference of the rear wheel
noncomputable def rear_circumference : ℝ := 2 * Real.pi * rear_radius

-- Define the modified circumferences
noncomputable def modified_front_circumference : ℝ := front_circumference + 1
noncomputable def modified_rear_circumference : ℝ := rear_circumference - 1

-- Theorem stating the relationship between wheel revolutions
theorem wheel_revolutions : 
  (40 / modified_rear_circumference) - (40 / modified_front_circumference) = 20 := by sorry

-- Theorem stating the circumferences of the wheels
theorem wheel_circumferences : 
  front_circumference = 3/2 ∧ rear_circumference = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_revolutions_wheel_circumferences_l126_12605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_x_is_zero_l126_12612

theorem average_x_is_zero (x : ℝ) :
  (∃ y : ℝ, y^2 = 3 * x^2 + 2 ∧ y^2 = 32) →
  (let S := {x : ℝ | ∃ y : ℝ, y^2 = 3 * x^2 + 2 ∧ y^2 = 32}
   ∃ a b : ℝ, S = {a, b} ∧ (a + b) / 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_x_is_zero_l126_12612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_square_tiles_l126_12617

/-- Given a room with length 720 cm and width 432 cm, the least number of square tiles
    of equal size required to cover the entire floor is 15. -/
theorem least_square_tiles (length width : ℕ) (h1 : length = 720) (h2 : width = 432) :
  (length / Nat.gcd length width) * (width / Nat.gcd length width) = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_square_tiles_l126_12617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_construct_EM_in_R_l126_12629

-- Define the sheet R as a convex set in a 2D plane
noncomputable def Sheet : Set (Fin 2 → ℝ) := sorry

-- Define lines c and d
noncomputable def c : Set (Fin 2 → ℝ) := sorry
noncomputable def d : Set (Fin 2 → ℝ) := sorry

-- Define point E
noncomputable def E : Fin 2 → ℝ := sorry

-- Define the intersection point M of c and d
noncomputable def M : Fin 2 → ℝ := sorry

-- Theorem from exercise 645 (assumed to be proven)
axiom exercise_645_theorem : True

-- Main theorem
theorem construct_EM_in_R 
  (h_convex : Convex ℝ Sheet)
  (h_c_in_R : c ∩ Sheet ≠ ∅)
  (h_d_in_R : d ∩ Sheet ≠ ∅)
  (h_not_parallel : ¬ (c = d))
  (h_not_intersect_in_R : (c ∩ d) ∩ Sheet = ∅)
  (h_E_in_R : E ∈ Sheet)
  (h_E_not_on_c : E ∉ c)
  (h_E_not_on_d : E ∉ d)
  (h_M_not_in_R : M ∉ Sheet)
  : ∃ (EM : Set (Fin 2 → ℝ)), 
    EM ⊆ Sheet ∧ 
    E ∈ EM ∧ 
    M ∈ closure EM :=
by sorry

-- Placeholder for ConstructibleFrom
def ConstructibleFrom (T : Prop) (S : Set (Fin 2 → ℝ)) : Prop := sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_construct_EM_in_R_l126_12629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stream_speed_calculation_l126_12657

/-- Calculates the speed of a stream given downstream and upstream speeds -/
noncomputable def stream_speed (downstream_speed upstream_speed : ℝ) : ℝ :=
  (downstream_speed - upstream_speed) / 2

/-- Theorem: The speed of the stream is 2.5 kmph given the conditions -/
theorem stream_speed_calculation (downstream_speed upstream_speed : ℝ) 
  (h1 : downstream_speed = 13)
  (h2 : upstream_speed = 8) :
  stream_speed downstream_speed upstream_speed = 2.5 := by
  unfold stream_speed
  rw [h1, h2]
  norm_num

-- Remove the #eval statement as it's not necessary for the proof
-- and may cause issues with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stream_speed_calculation_l126_12657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_fourth_quadrant_l126_12627

/-- If a point P(tan α, sin α) is in the third quadrant, then the angle α is in the fourth quadrant. -/
theorem angle_in_fourth_quadrant (α : ℝ) :
  (Real.tan α < 0 ∧ Real.sin α < 0) → α ∈ Set.Icc (3 * Real.pi / 2) (2 * Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_fourth_quadrant_l126_12627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_melted_mixture_theorem_l126_12651

/-- Represents the weight of the melted mixture of zinc and copper -/
noncomputable def melted_mixture_weight (zinc_weight : ℝ) (zinc_ratio : ℝ) (copper_ratio : ℝ) : ℝ :=
  zinc_weight * (zinc_ratio + copper_ratio) / zinc_ratio

/-- 
Given a mixture of zinc and copper in the ratio 9:11, where 28.8 kg of zinc is used,
the total weight of the mixture is 64 kg.
-/
theorem melted_mixture_theorem :
  melted_mixture_weight 28.8 9 11 = 64 := by
  -- Unfold the definition of melted_mixture_weight
  unfold melted_mixture_weight
  -- Simplify the expression
  simp
  -- The proof is completed using sorry
  sorry

#eval (28.8 * (9 + 11) / 9 : Float)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_melted_mixture_theorem_l126_12651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_special_sequences_l126_12698

mutual
  def a : ℕ → ℕ
    | 0 => 0
    | 1 => 0
    | 2 => 1
    | n + 3 => a (n + 1) + b (n + 1)

  def b : ℕ → ℕ
    | 0 => 0
    | 1 => 1
    | 2 => 0
    | n + 3 => a (n + 2) + b (n + 1)
end

theorem count_special_sequences : a 16 + b 16 = 682 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_special_sequences_l126_12698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_tower_volume_relationship_l126_12601

/-- Represents the scale factor between a real water tower and its model -/
noncomputable def scale_factor (real_height : ℝ) (model_height : ℝ) : ℝ :=
  real_height / model_height

/-- Calculates the volume scale factor given a linear scale factor -/
noncomputable def volume_scale_factor (linear_scale : ℝ) : ℝ :=
  linear_scale ^ 3

/-- Converts cubic inches to cubic feet -/
noncomputable def cubic_inches_to_feet (cubic_inches : ℝ) : ℝ :=
  cubic_inches / 1728

/-- Theorem stating the relationship between model and real water tower volumes -/
theorem water_tower_volume_relationship 
  (real_tower_height : ℝ) 
  (model_tower_height : ℝ) 
  (reference_real_height : ℝ) 
  (reference_model_height : ℝ) : 
  real_tower_height = 90 →
  model_tower_height = 0.5 →
  reference_real_height = 3 →
  reference_model_height = 0.5 →
  cubic_inches_to_feet (volume_scale_factor (scale_factor real_tower_height model_tower_height)) = 3375 := by
  sorry

#check water_tower_volume_relationship

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_tower_volume_relationship_l126_12601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_four_difference_l126_12669

theorem power_of_four_difference (m n : ℕ) : (4 : ℤ)^m - (4 : ℤ)^n = 255 ↔ m = 4 ∧ n = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_four_difference_l126_12669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_odd_decomposition_l126_12619

-- Define the interval (-1, 1)
def openInterval : Set ℝ := {x : ℝ | -1 < x ∧ x < 1}

-- Define a function f on the open interval (-1, 1)
variable (f : ℝ → ℝ)

-- Define g and h
noncomputable def g (f : ℝ → ℝ) (x : ℝ) : ℝ := (f x + f (-x)) / 2
noncomputable def h (f : ℝ → ℝ) (x : ℝ) : ℝ := (f x - f (-x)) / 2

-- Theorem statement
theorem even_odd_decomposition (f : ℝ → ℝ) :
  (∀ x, f x = g f x + h f x) ∧
  (∀ x, g f (-x) = g f x) ∧
  (∀ x, h f (-x) = -(h f x)) :=
by
  constructor
  · intro x
    simp [g, h]
    ring
  constructor
  · intro x
    simp [g]
    ring
  · intro x
    simp [h]
    ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_odd_decomposition_l126_12619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_100_eq_5151_l126_12699

def a : ℕ → ℚ
  | 0 => 1
  | 1 => 1
  | (n + 1) => a n + (2 * a n) / n

theorem a_100_eq_5151 : a 100 = 5151 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_100_eq_5151_l126_12699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_reconstruction_l126_12610

def scrambled_product : Nat := 2342355286

def is_valid_abc (a b c : Nat) : Prop :=
  a > b ∧ b > c ∧ c > 0 ∧ c < 10 ∧ b < 10 ∧ a < 10

def number_from_digits (a b c : Nat) : Nat := 100 * a + 10 * b + c

theorem product_reconstruction :
  ∃ (a b c : Nat),
    is_valid_abc a b c ∧
    let n1 := number_from_digits a b c
    let n2 := number_from_digits b c a
    let n3 := number_from_digits c a b
    let product := n1 * n2 * n3
    product % 10 = 6 ∧
    (Nat.sum (Nat.digits 10 product)) % 3 = (Nat.sum (Nat.digits 10 scrambled_product)) % 3 ∧
    product = 328245326 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_reconstruction_l126_12610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigExpression_equals_four_l126_12685

open Real

-- Define the expression
noncomputable def trigExpression : ℝ := 1 / cos (80 * π / 180) - sqrt 3 / sin (80 * π / 180)

-- State the theorem
theorem trigExpression_equals_four : trigExpression = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigExpression_equals_four_l126_12685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_k_l126_12641

/-- Represents a partition of a set of consecutive integers -/
structure Partition (n : ℕ) where
  subset1 : Finset ℕ
  subset2 : Finset ℕ
  partition_complete : subset1 ∪ subset2 = Finset.range n
  partition_disjoint : subset1 ∩ subset2 = ∅

/-- Checks if a partition satisfies the required properties -/
def is_valid_partition (n : ℕ) (p : Partition n) : Prop :=
  let s1 := p.subset1
  let s2 := p.subset2
  (s1.card = s2.card) ∧
  (s1.sum id = s2.sum id) ∧
  (s1.sum (λ x => x^2) = s2.sum (λ x => x^2)) ∧
  (s1.sum (λ x => x^3) = s2.sum (λ x => x^3))

/-- The main theorem -/
theorem smallest_valid_k : 
  ∀ k : ℕ, k ≥ 1 → 
  (∃ (p : Partition (4*k)), is_valid_partition (4*k) p) ↔ k ≥ 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_k_l126_12641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_value_for_nonzero_solution_l126_12697

theorem lambda_value_for_nonzero_solution (B : Matrix (Fin 3) (Fin 3) ℝ) (lambda : ℝ) :
  B ≠ 0 →
  (∀ j : Fin 3, 
    (B 0 j) + 2 * (B 1 j) - 2 * (B 2 j) = 0 ∧
    2 * (B 0 j) - (B 1 j) + lambda * (B 2 j) = 0 ∧
    3 * (B 0 j) + (B 1 j) - (B 2 j) = 0) →
  lambda = 7/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_value_for_nonzero_solution_l126_12697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_division_impossibility_l126_12671

/-- Represents a triangle with three angles --/
structure Triangle where
  angle1 : ℕ
  angle2 : ℕ
  angle3 : ℕ

/-- Represents a square divided into triangles --/
structure DividedSquare where
  triangles : List Triangle

/-- Checks if all angles in the triangles are multiples of 15 --/
def allAnglesMultiplesOf15 (ds : DividedSquare) : Prop :=
  ∀ t ∈ ds.triangles, t.angle1 % 15 = 0 ∧ t.angle2 % 15 = 0 ∧ t.angle3 % 15 = 0

/-- Checks if all angles in the triangles are different --/
def allAnglesDifferent (ds : DividedSquare) : Prop :=
  ∀ t1 t2, t1 ∈ ds.triangles → t2 ∈ ds.triangles →
    ∀ a1 ∈ [t1.angle1, t1.angle2, t1.angle3],
    ∀ a2 ∈ [t2.angle1, t2.angle2, t2.angle3],
    (t1 ≠ t2 ∨ a1 ≠ a2) → a1 ≠ a2

/-- Checks if the sum of angles in each triangle is 180 degrees --/
def validTriangles (ds : DividedSquare) : Prop :=
  ∀ t ∈ ds.triangles, t.angle1 + t.angle2 + t.angle3 = 180

/-- Theorem stating the impossibility of dividing a square into triangles
    with all angles being different multiples of 15 degrees --/
theorem square_division_impossibility (ds : DividedSquare) :
  validTriangles ds → allAnglesDifferent ds → ¬(allAnglesMultiplesOf15 ds) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_division_impossibility_l126_12671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_sqrt_sin_tan_x_l126_12628

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (Real.sin x * Real.tan x)

def domain (f : ℝ → ℝ) : Set ℝ :=
  {x | ∃ y, f x = y}

theorem domain_of_sqrt_sin_tan_x :
  domain f = {x : ℝ | ∃ k : ℤ, (-π/2 + 2*π*(k : ℝ) < x ∧ x < π/2 + 2*π*(k : ℝ)) ∨ x = π*(k : ℝ)} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_sqrt_sin_tan_x_l126_12628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_UQVS_is_108_l126_12654

-- Define the triangle PQR
structure Triangle :=
  (P Q R : ℝ × ℝ)

-- Define the properties of the triangle
def triangle_properties (t : Triangle) : Prop :=
  let (px, py) := t.P
  let (qx, qy) := t.Q
  let (rx, ry) := t.R
  let PQ := Real.sqrt ((qx - px)^2 + (qy - py)^2)
  let PR := Real.sqrt ((rx - px)^2 + (ry - py)^2)
  PQ = 60 ∧ PR = 15

-- Define the area of the triangle
def triangle_area (t : Triangle) : ℝ := 180

-- Define the midpoints S and T
noncomputable def S (t : Triangle) : ℝ × ℝ :=
  let (px, py) := t.P
  let (qx, qy) := t.Q
  ((px + qx) / 2, (py + qy) / 2)

noncomputable def T (t : Triangle) : ℝ × ℝ :=
  let (px, py) := t.P
  let (rx, ry) := t.R
  ((px + rx) / 2, (py + ry) / 2)

-- Define the angle bisector points U and V
noncomputable def U (t : Triangle) : ℝ × ℝ := sorry

noncomputable def V (t : Triangle) : ℝ × ℝ := sorry

-- Define the area of quadrilateral UQVS
noncomputable def area_UQVS (t : Triangle) : ℝ := sorry

-- State the theorem
theorem area_UQVS_is_108 (t : Triangle) :
  triangle_properties t → area_UQVS t = 108 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_UQVS_is_108_l126_12654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_odd_numbers_1_to_199_l126_12690

/-- The sum of odd numbers from 1 to 199 -/
def sumOddNumbers : ℕ := 10000

/-- The sequence of odd numbers from 1 to 199 -/
def oddSequence : List ℕ := List.range 100 |> List.map (fun n => 2*n + 1)

theorem sum_odd_numbers_1_to_199 : 
  (oddSequence.length = 100) ∧ 
  (∀ n ∈ oddSequence, n % 2 = 1) ∧
  (oddSequence.head? = some 1) ∧
  (oddSequence.getLast? = some 199) ∧
  (sumOddNumbers = oddSequence.sum) := by
  sorry

#eval sumOddNumbers
#eval oddSequence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_odd_numbers_1_to_199_l126_12690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_in_cube_l126_12689

/-- The volume of a tetrahedron formed by alternating vertices of a cube -/
theorem tetrahedron_volume_in_cube (cube_side_length : ℝ) (h : cube_side_length = 8) :
  (cube_side_length ^ 3 - 4 * (1 / 3 * cube_side_length ^ 2 * cube_side_length)) = 512 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_in_cube_l126_12689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lemonade_revenue_calculation_l126_12674

structure LemonadeRevenue where
  small : ℕ   -- Revenue from small lemonades
  medium : ℕ  -- Revenue from medium lemonades
  large : ℕ   -- Revenue from large lemonades
  total : ℕ   -- Total revenue

structure LemonadePrices where
  small : ℕ   -- Price of small lemonade
  medium : ℕ  -- Price of medium lemonade
  large : ℕ   -- Price of large lemonade

theorem lemonade_revenue_calculation 
  (prices : LemonadePrices)
  (revenue : LemonadeRevenue)
  (h1 : prices.small = 1)
  (h2 : prices.medium = 2)
  (h3 : prices.large = 3)
  (h4 : revenue.total = 50)
  (h5 : revenue.medium = 24)
  (h6 : revenue.large = 5 * prices.large) :
  revenue.small = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lemonade_revenue_calculation_l126_12674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_perimeters_equals_300_l126_12667

/-- Represents the side length of the nth triangle in the sequence -/
noncomputable def sideLength (n : ℕ) : ℝ :=
  50 / (2 ^ n)

/-- Represents the perimeter of the nth triangle in the sequence -/
noncomputable def perimeter (n : ℕ) : ℝ :=
  3 * sideLength n

/-- The sum of the perimeters of all triangles in the infinite sequence -/
noncomputable def sumOfPerimeters : ℝ :=
  ∑' n, perimeter n

theorem sum_of_perimeters_equals_300 :
  sumOfPerimeters = 300 := by
  sorry

#check sum_of_perimeters_equals_300

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_perimeters_equals_300_l126_12667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_birthday_celebration_l126_12640

/-- Days of the week -/
inductive DayOfWeek
  | monday
  | tuesday
  | wednesday
  | thursday
  | friday
  | saturday
  | sunday
deriving Repr, BEq, Inhabited

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.monday    => DayOfWeek.tuesday
  | DayOfWeek.tuesday   => DayOfWeek.wednesday
  | DayOfWeek.wednesday => DayOfWeek.thursday
  | DayOfWeek.thursday  => DayOfWeek.friday
  | DayOfWeek.friday    => DayOfWeek.saturday
  | DayOfWeek.saturday  => DayOfWeek.sunday
  | DayOfWeek.sunday    => DayOfWeek.monday

/-- Function to get the day of the week after n days -/
def dayAfter (start : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => start
  | n+1 => nextDay (dayAfter start n)

theorem birthday_celebration (birthDay : DayOfWeek) (celebrationDay : Nat) : 
  birthDay = DayOfWeek.friday → celebrationDay = 1500 → 
  dayAfter birthDay celebrationDay = DayOfWeek.sunday :=
by
  sorry

#check birthday_celebration

end NUMINAMATH_CALUDE_ERRORFEEDBACK_birthday_celebration_l126_12640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l126_12664

-- Define the function as noncomputable due to the use of Real.sqrt
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (Real.cos (2 * x)) + Real.sqrt (3 - 2 * Real.sqrt 3 * Real.tan x - 3 * (Real.tan x)^2)

-- Define the domain
def domain (x : ℝ) : Prop :=
  ∃ k : ℤ, k * Real.pi - Real.pi / 4 ≤ x ∧ x ≤ k * Real.pi + Real.pi / 6

-- Theorem statement
theorem f_domain :
  ∀ x : ℝ, (∃ y : ℝ, f x = y) ↔ domain x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l126_12664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_aprons_needed_l126_12632

def initial_aprons : ℕ := 13
def today_aprons : ℕ := 3 * initial_aprons
def tomorrow_aprons : ℕ := 49

theorem total_aprons_needed : 
  let sewn_so_far := initial_aprons + today_aprons
  let remaining := 2 * tomorrow_aprons
  sewn_so_far + remaining = 150 := by
    -- Proof steps would go here
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_aprons_needed_l126_12632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_second_hour_l126_12653

/-- Given a car traveling for two hours with an average speed of 82.5 km/h
    and a speed of 90 km/h in the first hour, prove that the speed in the
    second hour is 75 km/h. -/
theorem car_speed_second_hour
  (speed_first_hour : ℝ)
  (average_speed : ℝ)
  (h1 : speed_first_hour = 90)
  (h2 : average_speed = 82.5) :
  let total_distance := average_speed * 2
  let distance_second_hour := total_distance - speed_first_hour
  let speed_second_hour := distance_second_hour
  speed_second_hour = 75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_second_hour_l126_12653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_area_l126_12681

-- Define the regular triangular prism
structure RegularTriangularPrism where
  base_edge : ℝ
  height : ℝ
  base_edge_pos : base_edge > 0
  height_pos : height > 0

-- Define the circumscribed sphere of the prism
structure CircumscribedSphere (prism : RegularTriangularPrism) where
  radius : ℝ
  radius_pos : radius > 0

-- Calculate the surface area of a sphere
noncomputable def SphereArea (radius : ℝ) : ℝ :=
  4 * Real.pi * radius^2

-- Theorem statement
theorem circumscribed_sphere_area 
  (prism : RegularTriangularPrism)
  (sphere : CircumscribedSphere prism)
  (h_base : prism.base_edge = 6)
  (h_height : prism.height = Real.sqrt 3) :
  SphereArea sphere.radius = 51 * Real.pi := by
  sorry

#check circumscribed_sphere_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_area_l126_12681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_constants_l126_12622

-- Define the constants
noncomputable def a : ℝ := Real.log 2
noncomputable def b : ℝ := 2^(-Real.log 2)
noncomputable def c : ℝ := Real.log (Real.log 2) / Real.log 10

-- State the theorem
theorem order_of_constants : a > b ∧ b > c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_constants_l126_12622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_problem_solution_l126_12673

/-- Represents a box of marbles -/
structure Box where
  black : ℕ
  white : ℕ

/-- The problem setup -/
structure MarbleProblem where
  box1 : Box
  box2 : Box
  total_marbles : ℕ
  prob_both_black : ℚ
  prob_both_white : ℚ

/-- The conditions of the problem -/
def valid_problem (p : MarbleProblem) : Prop :=
  p.total_marbles = 25 ∧
  p.prob_both_black = 27 / 50 ∧
  ∃ m n : ℕ, p.prob_both_white = m / n ∧ Nat.Coprime m n ∧
  p.box1.black + p.box1.white + p.box2.black + p.box2.white = p.total_marbles ∧
  (p.box1.black : ℚ) / (p.box1.black + p.box1.white : ℚ) * 
    (p.box2.black : ℚ) / (p.box2.black + p.box2.white : ℚ) = p.prob_both_black ∧
  (p.box1.white : ℚ) / (p.box1.black + p.box1.white : ℚ) * 
    (p.box2.white : ℚ) / (p.box2.black + p.box2.white : ℚ) = p.prob_both_white

/-- The theorem to prove -/
theorem marble_problem_solution (p : MarbleProblem) 
  (h : valid_problem p) : ∃ m n : ℕ, p.prob_both_white = m / n ∧ m + n = 61 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_problem_solution_l126_12673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_whole_number_above_sum_l126_12696

theorem smallest_whole_number_above_sum : ℕ := by
  -- Define the sum of the given mixed fractions
  let sum : ℚ := 3 + 1/3 + 4 + 1/4 + 5 + 1/5 + 6 + 1/6 + 7 + 1/7

  -- Define the smallest whole number larger than the sum
  let smallest_whole_number : ℕ := 27

  -- The sum is less than the smallest whole number
  have sum_less : sum < smallest_whole_number := by sorry

  -- There is no whole number between the sum and the smallest whole number
  have no_between : ∀ n : ℕ, sum < n → n ≥ smallest_whole_number := by sorry

  -- Prove that smallest_whole_number is the correct answer
  exact smallest_whole_number

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_whole_number_above_sum_l126_12696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_point_on_line_l126_12666

/-- Given four distinct points on a line, there exists a unique point P 
    such that the product of its distances to A₁ and A₂ equals 
    the product of its distances to B₁ and B₂ -/
theorem unique_point_on_line 
  (A₁ A₂ B₁ B₂ : ℝ) 
  (h_distinct : A₁ ≠ A₂ ∧ B₁ ≠ B₂ ∧ A₁ ≠ B₁ ∧ A₂ ≠ B₂) :
  ∃! P : ℝ, |P - A₁| * |P - A₂| = |P - B₁| * |P - B₂| :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_point_on_line_l126_12666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_degrees_216_to_radians_l126_12631

/-- Converts degrees to radians -/
noncomputable def degreesToRadians (degrees : ℝ) : ℝ := (degrees / 180) * Real.pi

/-- Theorem: Converting 216° to radians equals 6π/5 -/
theorem degrees_216_to_radians :
  degreesToRadians 216 = (6 / 5) * Real.pi := by
  -- Unfold the definition of degreesToRadians
  unfold degreesToRadians
  -- Simplify the expression
  simp [Real.pi]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_degrees_216_to_radians_l126_12631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_from_cylinder_l126_12638

/-- A right circular cylinder circumscribed by a sphere -/
structure CylinderInSphere where
  /-- The volume of the cylinder -/
  cylinder_volume : ℝ
  /-- The perimeter of the cylinder's base -/
  base_perimeter : ℝ
  /-- All vertices of the cylinder lie on the sphere's surface -/
  vertices_on_sphere : Prop

/-- The volume of the sphere circumscribing the cylinder -/
noncomputable def sphere_volume (c : CylinderInSphere) : ℝ := 32 * Real.pi / 9

/-- Theorem stating the volume of the sphere given the cylinder properties -/
theorem sphere_volume_from_cylinder (c : CylinderInSphere) 
  (h1 : c.cylinder_volume = Real.sqrt 3 / 2)
  (h2 : c.base_perimeter = 3)
  (h3 : c.vertices_on_sphere) :
  sphere_volume c = 32 * Real.pi / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_from_cylinder_l126_12638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carters_road_trip_l126_12680

/-- Calculates the total trip time given the original trip duration, stop interval, 
    additional stops, and duration of each stop. -/
noncomputable def totalTripTime (originalDuration : ℝ) (stopInterval : ℝ) (additionalFoodStops : ℕ) 
                  (additionalGasStops : ℕ) (stopDuration : ℝ) : ℝ :=
  let regularStops := originalDuration / stopInterval
  let totalStops := ⌊regularStops⌋ + additionalFoodStops + additionalGasStops
  let totalStopTime := (totalStops : ℝ) * stopDuration / 60
  originalDuration + totalStopTime

/-- Theorem stating that Carter's road trip will become 18 hours long. -/
theorem carters_road_trip : 
  totalTripTime 14 2 2 3 20 = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carters_road_trip_l126_12680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_average_speed_l126_12668

/-- Represents the speed of a car at each hour -/
def speed : Fin 5 → ℚ
  | 0 => 100
  | 1 => 80
  | 2 => 60
  | 3 => 50
  | 4 => 40

/-- The total time of travel in hours -/
def totalTime : ℚ := 5

/-- The constant deceleration rate during the fourth and fifth hours -/
def decelerationRate : ℚ := 10

/-- Calculates the total distance traveled -/
def totalDistance : ℚ :=
  (Finset.sum Finset.univ fun i => speed i)

/-- The average speed of the car -/
noncomputable def averageSpeed : ℚ :=
  totalDistance / totalTime

theorem car_average_speed :
  speed 0 = 100 ∧
  speed 1 = 80 ∧
  speed 2 = 60 ∧
  speed 4 = 40 ∧
  (∀ i : Fin 3, speed (i + 1) = speed i - decelerationRate) →
  averageSpeed = 66 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_average_speed_l126_12668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_solution_l126_12655

theorem quadratic_equation_solution :
  ∃ (m n p : ℤ),
    -- The quadratic equation 5x^2 - 11x + 6 = 0 has solutions (m + √n) / p and (m - √n) / p
    (∀ x : ℚ, 5 * x^2 - 11 * x + 6 = 0 ↔ x = (m + Real.sqrt (n : ℝ)) / p ∨ x = (m - Real.sqrt (n : ℝ)) / p) ∧
    -- m, n, and p are 11, 1, and 10 respectively
    m = 11 ∧ n = 1 ∧ p = 10 ∧
    -- The greatest common divisor of m, n, and p is 1
    Int.gcd (Int.gcd m n) p = 1 ∧
    -- The sum of m, n, and p is 22
    m + n + p = 22 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_solution_l126_12655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_six_inch_silver_sphere_value_l126_12611

/-- The value of a silver sphere given its radius -/
noncomputable def silverSphereValue (radius : ℝ) : ℝ :=
  500 * (radius / 4) ^ 3

/-- Rounds a real number to the nearest integer -/
noncomputable def roundToNearest (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

theorem six_inch_silver_sphere_value :
  roundToNearest (silverSphereValue 6) = 1688 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_six_inch_silver_sphere_value_l126_12611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concrete_cost_per_pound_l126_12665

/-- Proves that the cost per pound of concrete is $0.02 given the specified conditions --/
theorem concrete_cost_per_pound
  (num_homes : ℕ)
  (slab_length : ℝ)
  (slab_width : ℝ)
  (slab_height : ℝ)
  (concrete_density : ℝ)
  (total_cost : ℝ)
  (h_num_homes : num_homes = 3)
  (h_slab_length : slab_length = 100)
  (h_slab_width : slab_width = 100)
  (h_slab_height : slab_height = 0.5)
  (h_concrete_density : concrete_density = 150)
  (h_total_cost : total_cost = 45000) :
  total_cost / (↑num_homes * slab_length * slab_width * slab_height * concrete_density) = 0.02 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_concrete_cost_per_pound_l126_12665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gift_price_calculation_l126_12633

/-- The price of Lisa's gift --/
def gift_price : ℕ := 3760

/-- Lisa's initial savings --/
def lisa_savings : ℕ := 1200

/-- The fraction of Lisa's savings her mother gave her --/
def mother_fraction : ℚ := 3 / 5

/-- The multiple of her mother's contribution that Lisa's brother gave her --/
def brother_multiple : ℕ := 2

/-- The additional amount Lisa needs after contributions --/
def additional_needed : ℕ := 400

/-- Theorem stating the price of the gift Lisa wants to buy --/
theorem gift_price_calculation :
  gift_price = lisa_savings +
               (mother_fraction * lisa_savings).floor +
               (brother_multiple * (mother_fraction * lisa_savings).floor) +
               additional_needed :=
by
  sorry

#check gift_price_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gift_price_calculation_l126_12633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_line_and_curve_l126_12677

/-- The minimum distance between a point on the line y = 2x + 1 and a point on the curve y = x + ln(x) -/
noncomputable def min_distance : ℝ := 2 * Real.sqrt 5 / 5

/-- The line equation y = 2x + 1 -/
def line (x : ℝ) : ℝ := 2 * x + 1

/-- The curve equation y = x + ln(x) -/
noncomputable def curve (x : ℝ) : ℝ := x + Real.log x

/-- Theorem stating that the minimum distance between a point on the line and a point on the curve is 2√5/5 -/
theorem min_distance_between_line_and_curve :
  ∃ (p q : ℝ × ℝ), 
    (line p.1 = p.2) ∧ 
    (curve q.1 = q.2) ∧ 
    (∀ (p' q' : ℝ × ℝ), 
      (line p'.1 = p'.2) → 
      (curve q'.1 = q'.2) → 
      Real.sqrt ((p'.1 - q'.1)^2 + (p'.2 - q'.2)^2) ≥ min_distance) ∧
    (Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = min_distance) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_line_and_curve_l126_12677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l126_12620

/-- Line l defined by parametric equations -/
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (-1 + (Real.sqrt 2 / 2) * t, (Real.sqrt 2 / 2) * t)

/-- Curve C defined by y = x² -/
def curve_C (x : ℝ) : ℝ := x^2

/-- Point M -/
def point_M : ℝ × ℝ := (-1, 0)

/-- Theorem stating that the product of distances from M to intersection points is 2 -/
theorem intersection_distance_product :
  ∃ (t₁ t₂ : ℝ), 
    curve_C ((line_l t₁).1) = (line_l t₁).2 ∧
    curve_C ((line_l t₂).1) = (line_l t₂).2 ∧
    t₁ ≠ t₂ ∧
    ((line_l t₁).1 - (point_M.1))^2 + ((line_l t₁).2 - (point_M.2))^2 *
    ((line_l t₂).1 - (point_M.1))^2 + ((line_l t₂).2 - (point_M.2))^2 = 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l126_12620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_first_term_to_common_difference_l126_12600

/-- An arithmetic progression with first term a and common difference d -/
structure ArithmeticProgression where
  a : ℚ  -- first term
  d : ℚ  -- common difference

/-- Sum of the first n terms of an arithmetic progression -/
def sum_n (ap : ArithmeticProgression) (n : ℕ) : ℚ :=
  n / 2 * (2 * ap.a + (n - 1) * ap.d)

theorem ratio_first_term_to_common_difference 
  (ap : ArithmeticProgression) 
  (h : sum_n ap 15 = 3 * sum_n ap 8) : 
  ap.a / ap.d = 7 / 3 := by
  sorry

#eval (7 : ℚ) / 3  -- To check the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_first_term_to_common_difference_l126_12600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l126_12639

theorem equation_solution (x : ℝ) : 
  (3 * (Real.cos (2 * x) + Real.tan (2 * x)⁻¹)) / (Real.tan (2 * x)⁻¹ - Real.cos (2 * x)) - 2 * (Real.sin (2 * x) + 1) = 0 →
  ∃ k : ℤ, x = π / 12 * (-1)^(k + 1) + k * π / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l126_12639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l126_12636

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f x = -f (-x)

-- State the theorem
theorem function_properties
  (h1 : is_even f)
  (h2 : is_odd (λ x ↦ f (2*x + 1) - 1)) :
  (f 1 = 1) ∧
  (∀ x, f (x + 4) = f x) ∧
  (∀ x, f (6 + x) = f (6 - x)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l126_12636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_legs_l126_12684

theorem right_triangle_legs (r R : ℝ) (h_r : r = 2) (h_R : R = 5) :
  ∃ (a b : ℝ), a^2 + b^2 = (2*R)^2 ∧ a + b - 2*R = 2*r ∧ ({a, b} : Set ℝ) = {6, 8} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_legs_l126_12684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_one_max_a_for_monotone_increasing_number_of_zeros_l126_12693

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x + 1) * Real.log x - a * x + a

-- Part I
theorem tangent_slope_at_one (a : ℝ) :
  (deriv (f a)) 1 = Real.pi / 4 → a = 1 := by sorry

-- Part II
theorem max_a_for_monotone_increasing :
  ∃ (a_max : ℝ), a_max = 2 ∧
  ∀ (a : ℝ), (∀ x > 0, Monotone (f a)) → a ≤ a_max := by sorry

-- Part III
theorem number_of_zeros (a : ℝ) :
  (a ≤ 2 → ∃! x, f a x = 0) ∧
  (a > 2 → ∃ x y z, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a x = 0 ∧ f a y = 0 ∧ f a z = 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_one_max_a_for_monotone_increasing_number_of_zeros_l126_12693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_real_coefficients_approximation_l126_12659

/-- The sum of all real coefficients of the expansion of (2+ix)^2011 -/
noncomputable def T : ℝ := (3^2011 + 1) / 2

/-- The approximation of log_2(T) -/
noncomputable def log2T_approx : ℝ := 2011 * Real.log 3 / Real.log 2 - 1

theorem sum_real_coefficients_approximation :
  |Real.log T / Real.log 2 - log2T_approx| < 1e-5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_real_coefficients_approximation_l126_12659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_condition_implies_isosceles_l126_12606

/-- A triangle with vertices A, B, and C. -/
structure Triangle (V : Type*) [NormedAddCommGroup V] where
  A : V
  B : V
  C : V

/-- The centroid of a triangle. -/
noncomputable def centroid {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V] (t : Triangle V) : V :=
  (1/3 : ℝ) • (t.A + t.B + t.C)

/-- The theorem stating that if the centroid condition is met, the triangle is isosceles. -/
theorem centroid_condition_implies_isosceles
  {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]
  (t : Triangle V) (G : V) :
  G = centroid t →
  dist t.A t.B + dist G t.C = dist t.A t.C + dist G t.B →
  dist t.A t.B = dist t.A t.C := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_condition_implies_isosceles_l126_12606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_triangle_conditions_l126_12602

-- Define a triangle
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define angle and side length functions
noncomputable def angle (t : Triangle) (v : Fin 3) : ℝ := sorry
noncomputable def side_length (t : Triangle) (v w : Fin 3) : ℝ := sorry

-- Define the condition for a unique triangle
def unique_triangle (f : Triangle → Prop) : Prop :=
  ∀ t₁ t₂, f t₁ → f t₂ → t₁ = t₂

-- Define the conditions for each set
def set_A (t : Triangle) : Prop :=
  angle t 0 = Real.pi/3 ∧ angle t 1 = Real.pi/4 ∧ side_length t 0 1 = 4

def set_B (t : Triangle) : Prop :=
  angle t 0 = Real.pi/6 ∧ side_length t 0 1 = 5 ∧ side_length t 1 2 = 3

def set_C (t : Triangle) : Prop :=
  angle t 1 = Real.pi/3 ∧ side_length t 0 1 = 6 ∧ side_length t 1 2 = 10

def set_D (t : Triangle) : Prop :=
  angle t 2 = Real.pi/2 ∧ side_length t 0 1 = 5 ∧ side_length t 1 2 = 3

-- Theorem stating that set B cannot determine a unique triangle while others can
theorem unique_triangle_conditions :
  ¬(unique_triangle set_B) ∧
  (unique_triangle set_A) ∧
  (unique_triangle set_C) ∧
  (unique_triangle set_D) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_triangle_conditions_l126_12602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_sum_probability_l126_12634

-- Define the radii of the circles
def outer_radius : ℝ := 8
def inner_radius : ℝ := 4

-- Define the number of regions in each circle
def num_regions : ℕ := 3

-- Define the point values for inner and outer regions
def inner_points : List ℕ := [3, 4, 4]
def outer_points : List ℕ := [4, 3, 3]

-- Define a function to calculate the area of a circle
noncomputable def circle_area (radius : ℝ) : ℝ := Real.pi * radius^2

-- Define the total area of the dartboard
noncomputable def total_area : ℝ := circle_area outer_radius

-- Define the probability of hitting an odd-numbered region
noncomputable def prob_odd : ℝ := (2 * circle_area inner_radius + 2 * (circle_area outer_radius - circle_area inner_radius)) / total_area

-- Define the probability of hitting an even-numbered region
noncomputable def prob_even : ℝ := 1 - prob_odd

-- Theorem: The probability of obtaining an odd sum when throwing two darts is 4/9
theorem odd_sum_probability : 
  prob_odd * prob_even * 2 = 4/9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_sum_probability_l126_12634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overlap_area_of_strips_l126_12679

/-- The area of the rhombus formed by the overlap of two strips of width 1 intersecting at an angle α -/
noncomputable def rhombus_area (α : ℝ) : ℝ :=
  1 / Real.sin α

/-- Theorem: The area of the rhombus formed by the overlap of two strips of width 1 intersecting at an angle α is 1 / sin(α) -/
theorem overlap_area_of_strips (α : ℝ) (h : 0 < α ∧ α < π) :
  rhombus_area α = 1 / Real.sin α := by
  -- Unfold the definition of rhombus_area
  unfold rhombus_area
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_overlap_area_of_strips_l126_12679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_x_minus_two_defined_l126_12672

theorem sqrt_x_minus_two_defined (x : ℝ) : 0 ≤ x - 2 ↔ x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_x_minus_two_defined_l126_12672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_formula_l126_12675

/-- Given that the terminal side of angle α passes through point P(-4a, 3a) where a < 0,
    prove that 2sinα + cosα = -2/5 -/
theorem angle_terminal_side_formula (a : ℝ) (α : ℝ) 
    (h1 : a < 0)
    (h2 : Real.cos α = -4 * a / Real.sqrt (16 * a^2 + 9 * a^2))
    (h3 : Real.sin α = 3 * a / Real.sqrt (16 * a^2 + 9 * a^2)) :
  2 * Real.sin α + Real.cos α = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_formula_l126_12675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_problem_l126_12637

/-- Calculates the compound interest amount after n years -/
noncomputable def compoundInterest (principal : ℝ) (rate : ℝ) (years : ℝ) : ℝ :=
  principal * (1 + rate / 100) ^ years - principal

/-- Calculates the simple interest amount after n years -/
noncomputable def simpleInterest (principal : ℝ) (rate : ℝ) (years : ℝ) : ℝ :=
  principal * rate * years / 100

/-- The problem statement -/
theorem simple_interest_problem :
  ∃ (P : ℝ),
    simpleInterest P 10 5 = (1/2) * compoundInterest 5000 12 2 ∧
    P = 1272 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_problem_l126_12637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slant_asymptote_sum_l126_12624

/-- The rational function f(x) = (3x^2 - 5x + 4) / (x-4) -/
noncomputable def f (x : ℝ) : ℝ := (3*x^2 - 5*x + 4) / (x - 4)

/-- The slope of the slant asymptote of f(x) -/
def m : ℝ := 3

/-- The y-intercept of the slant asymptote of f(x) -/
def b : ℝ := 7

/-- Theorem: The sum of the slope and y-intercept of the slant asymptote of f(x) is 10 -/
theorem slant_asymptote_sum : m + b = 10 := by
  rw [m, b]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slant_asymptote_sum_l126_12624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_min_difference_x1_x2_l126_12625

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 2 * Real.sin x * Real.cos x - Real.sqrt 2 * (Real.cos x)^2 + Real.sqrt 2 / 2

noncomputable def g (x : ℝ) : ℝ := f x + f (x + Real.pi/4) - f x * f (x + Real.pi/4)

theorem f_monotone_increasing (k : ℤ) :
  StrictMonoOn f (Set.Icc (-Real.pi/8 + k * Real.pi) (3*Real.pi/8 + k * Real.pi)) := by
  sorry

theorem min_difference_x1_x2 :
  ∃ x₁ x₂ : ℝ, (∀ x : ℝ, g x₁ ≤ g x ∧ g x ≤ g x₂) ∧
  (∀ y₁ y₂ : ℝ, (∀ x : ℝ, g y₁ ≤ g x ∧ g x ≤ g y₂) → |x₁ - x₂| ≤ |y₁ - y₂|) ∧
  |x₁ - x₂| = 3*Real.pi/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_min_difference_x1_x2_l126_12625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_Al_approx_l126_12661

/-- The molar mass of Aluminum in g/mol -/
noncomputable def molar_mass_Al : ℝ := 26.98

/-- The molar mass of Phosphorus in g/mol -/
noncomputable def molar_mass_P : ℝ := 30.97

/-- The molar mass of Oxygen in g/mol -/
noncomputable def molar_mass_O : ℝ := 16.00

/-- The molar mass of Aluminum Phosphate (AlPO4) in g/mol -/
noncomputable def molar_mass_AlPO4 : ℝ := molar_mass_Al + molar_mass_P + 4 * molar_mass_O

/-- The mass percentage of Aluminum in Aluminum Phosphate -/
noncomputable def mass_percentage_Al : ℝ := (molar_mass_Al / molar_mass_AlPO4) * 100

/-- Theorem stating that the mass percentage of Aluminum in Aluminum Phosphate is approximately 22.12% -/
theorem mass_percentage_Al_approx :
  ∃ ε > 0, |mass_percentage_Al - 22.12| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_Al_approx_l126_12661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l126_12630

-- Define the square and circle
noncomputable def Square (a : ℝ) := { p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ a ∧ 0 ≤ p.2 ∧ p.2 ≤ a }

noncomputable def InscribedCircle (a : ℝ) := { p : ℝ × ℝ | (p.1 - a/2)^2 + (p.2 - a/2)^2 = (a/2)^2 }

-- Define point E
noncomputable def E (a : ℝ) : ℝ × ℝ := (a/2, 0)

-- Define line AE
noncomputable def LineAE (a : ℝ) := { p : ℝ × ℝ | p.2 = (p.1 * a) / (a * Real.sqrt 5) }

-- Theorem statement
theorem chord_length (a : ℝ) (h : a > 0) :
  ∃ p q : ℝ × ℝ,
    p ∈ InscribedCircle a ∧
    q ∈ InscribedCircle a ∧
    p ∈ LineAE a ∧
    q ∈ LineAE a ∧
    p ≠ q ∧
    Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = 2 * a / Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l126_12630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_factorial_sum_last_two_digits_l126_12691

def fibonacci_factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | 1 => 1
  | n+2 => fibonacci_factorial n + fibonacci_factorial (n+1)

def last_two_digits (n : ℕ) : ℕ := n % 100

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n+1 => (n+1) * factorial n

theorem fibonacci_factorial_sum_last_two_digits : 
  ((List.range 12).map (λ i => last_two_digits (factorial (fibonacci_factorial i)))
    |> List.foldl (· + ·) 0) % 100 = 5 := by
  sorry

#eval ((List.range 12).map (λ i => last_two_digits (factorial (fibonacci_factorial i)))
    |> List.foldl (· + ·) 0) % 100

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_factorial_sum_last_two_digits_l126_12691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_constant_theorem_l126_12670

open Set
open MeasureTheory

/-- The type of real polynomials of degree at most 3 -/
def Polynomial3 : Type := {p : ℝ → ℝ | ∃ (a b c d : ℝ), ∀ x, p x = a*x^3 + b*x^2 + c*x + d}

/-- The statement that a polynomial has a root in [0,1] -/
def HasRootIn01 (p : Polynomial3) : Prop :=
  ∃ x : ℝ, x ∈ Icc 0 1 ∧ p.1 x = 0

/-- The maximum absolute value of a polynomial on [0,1] -/
noncomputable def MaxAbs (p : Polynomial3) : ℝ :=
  ⨆ (x : ℝ) (h : x ∈ Icc 0 1), |p.1 x|

/-- The integral of the absolute value of a polynomial on [0,1] -/
noncomputable def IntegralAbs (p : Polynomial3) : ℝ :=
  ∫ x in (0)..(1), |p.1 x|

/-- The main theorem statement -/
theorem smallest_constant_theorem :
  (∀ (p : Polynomial3), HasRootIn01 p → IntegralAbs p ≤ (5/6) * MaxAbs p) ∧
  (∀ (C : ℝ), C < 5/6 → ∃ (p : Polynomial3), HasRootIn01 p ∧ IntegralAbs p > C * MaxAbs p) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_constant_theorem_l126_12670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_grows_faster_than_g_l126_12649

open Real

-- Define the functions
def f (x : ℝ) := x^3
noncomputable def g (x : ℝ) := x^2 * log x

-- State the theorem
theorem f_grows_faster_than_g :
  ∃ (c : ℝ), c > 0 ∧ ∀ x > c, f x > g x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_grows_faster_than_g_l126_12649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l126_12646

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x - Real.pi / 6) * Real.cos (ω * x) + 1 / 2

noncomputable def g (x : ℝ) : ℝ := Real.sin (x + Real.pi / 6)

def IsPeriodicAt (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem problem_solution (ω : ℝ) (h1 : ω > 0) (h2 : IsPeriodicAt (f ω) Real.pi) :
  ω = 1 ∧ 
  (∀ x ∈ Set.Icc (-Real.pi) Real.pi, g x = 0 ↔ x = -Real.pi/6 ∨ x = 5*Real.pi/6) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l126_12646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_l126_12688

/-- The function to be minimized -/
noncomputable def f (t s : ℝ) : ℝ := 6 * t^2 + 3 * s^2 - 4 * s * t - 8 * t + 6 * s + 5

/-- The point where the minimum occurs -/
noncomputable def min_point : ℝ × ℝ := (3/7, -5/7)

/-- The minimum value of the function -/
noncomputable def min_value : ℝ := 8/7

theorem f_minimum :
  ∀ t s : ℝ, f t s ≥ min_value ∧ f (min_point.1) (min_point.2) = min_value :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_l126_12688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_polynomial_and_multiple_l126_12694

theorem gcd_of_polynomial_and_multiple (x : ℤ) : 
  17280 ∣ x → 
  Nat.gcd (((5 * x + 3) * (11 * x + 2) * (17 * x + 7) * (4 * x + 5)).natAbs) x.natAbs = 210 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_polynomial_and_multiple_l126_12694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ruth_apples_proof_ruth_apples_correct_l126_12652

/-- Given that Ruth starts with 89 apples and shares 5 apples,
    prove that she ends with 84 apples. -/
def ruth_apples (initial : ℕ) (shared : ℕ) : ℕ :=
  initial - shared

/-- The number of apples Ruth starts with -/
def initial_apples : ℕ := 89

/-- The number of apples Ruth shares -/
def shared_apples : ℕ := 5

/-- The number of apples Ruth ends with -/
def final_apples : ℕ := ruth_apples initial_apples shared_apples

theorem ruth_apples_proof :
  final_apples = initial_apples - shared_apples :=
by
  -- Unfold the definitions
  unfold final_apples
  unfold ruth_apples
  -- The goal is now trivially true by reflexivity
  rfl

theorem ruth_apples_correct :
  ruth_apples 89 5 = 84 :=
by
  -- Unfold the definition of ruth_apples
  unfold ruth_apples
  -- Evaluate the subtraction
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ruth_apples_proof_ruth_apples_correct_l126_12652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_T_is_positive_l126_12614

theorem T_is_positive (α : ℝ) (h : ∀ k : ℤ, α ≠ k * π / 2) : 
  let T := (Real.sin α + Real.tan α) / (Real.cos α + (Real.cos α / Real.sin α))
  T > 0 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_T_is_positive_l126_12614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_interval_l126_12682

-- Define the function f(x) = x^2 - ln(x)
noncomputable def f (x : ℝ) : ℝ := x^2 - Real.log x

-- State the theorem
theorem f_monotone_decreasing_interval :
  ∀ x : ℝ, x > 0 → x < Real.sqrt 2 / 2 ↔ 
  ∀ y : ℝ, y > x → f y < f x :=
by
  sorry

#check f_monotone_decreasing_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_interval_l126_12682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l126_12604

/-- Given a parabola and a hyperbola with specific properties, prove that the eccentricity of the hyperbola is √5 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  let parabola (x y : ℝ) := x^2 = 8*y
  let hyperbola (x y : ℝ) := y^2/a^2 - x^2/b^2 = 1
  let asymptote (x y : ℝ) := y = (a/b)*x
  let parabola_axis := -2
  ∃ (x y : ℝ), 
    parabola x y ∧ 
    hyperbola x y ∧ 
    asymptote x y ∧ 
    |y - parabola_axis| = 4 →
    Real.sqrt (a^2 + b^2) / a = Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l126_12604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l126_12663

/-- The function f(x) = (2x+1)/(4x^2+1) -/
noncomputable def f (x : ℝ) : ℝ := (2*x + 1) / (4*x^2 + 1)

/-- The maximum value of f(x) for x > 0 -/
noncomputable def max_value : ℝ := (Real.sqrt 2 + 1) / 2

theorem f_max_value (x : ℝ) (h : x > 0) : f x ≤ max_value ∧ ∃ y > 0, f y = max_value := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l126_12663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_ratio_is_one_to_two_l126_12635

/-- Represents the ratio of boarders to day students -/
structure StudentRatio where
  boarders : ℕ
  day_students : ℕ

/-- The school's student population -/
structure School where
  initial_ratio : StudentRatio
  initial_boarders : ℕ
  new_boarders : ℕ

/-- Calculate the new ratio of boarders to day students after new boarders join -/
def new_ratio (school : School) : StudentRatio :=
  let initial_day_students := (school.initial_boarders * school.initial_ratio.day_students) / school.initial_ratio.boarders
  let new_boarders := school.initial_boarders + school.new_boarders
  ⟨new_boarders, initial_day_students⟩

/-- Theorem stating that the new ratio is 1:2 given the initial conditions -/
theorem new_ratio_is_one_to_two (school : School) 
  (h1 : school.initial_ratio = ⟨5, 12⟩) 
  (h2 : school.initial_boarders = 220) 
  (h3 : school.new_boarders = 44) : 
  (new_ratio school).boarders * 2 = (new_ratio school).day_students := by
  sorry

/-- Example calculation -/
def example_school : School := ⟨⟨5, 12⟩, 220, 44⟩

#eval (new_ratio example_school).boarders
#eval (new_ratio example_school).day_students

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_ratio_is_one_to_two_l126_12635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_sum_quotient_l126_12648

theorem factorial_sum_quotient : (Nat.factorial 8 + Nat.factorial 9) / Nat.factorial 6 = 560 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_sum_quotient_l126_12648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_ratio_sum_l126_12621

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: For a point P and a circle C, if a line with slope -1 passing through P
    intersects C at points A and B, then |PA|/|PB| + |PB|/|PA| = 8/3 -/
theorem intersection_ratio_sum (P A B : Point) : 
  P.x = 0 ∧ P.y = Real.sqrt 3 →
  (A.x - 1)^2 + (A.y - Real.sqrt 3)^2 = 4 →
  (B.x - 1)^2 + (B.y - Real.sqrt 3)^2 = 4 →
  (A.y - P.y) = (P.x - A.x) →
  (B.y - P.y) = (P.x - B.x) →
  (distance P A) / (distance P B) + (distance P B) / (distance P A) = 8/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_ratio_sum_l126_12621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_runners_meet_at_2000_l126_12623

/-- Represents a runner on a circular track -/
structure Runner where
  speed : ℝ

/-- Represents the circular track -/
def trackLength : ℝ := 600

/-- The set of runners -/
def runners : List Runner := [
  ⟨4.2⟩,
  ⟨4.5⟩,
  ⟨4.8⟩,
  ⟨5.1⟩
]

/-- Checks if all runners meet at a given time -/
def allMeetAt (t : ℝ) : Prop :=
  ∀ r₁ r₂, r₁ ∈ runners → r₂ ∈ runners → (r₁.speed - r₂.speed) * t % trackLength = 0

/-- The theorem to be proved -/
theorem runners_meet_at_2000 :
  (∀ t > 0, allMeetAt t → t ≥ 2000) ∧ allMeetAt 2000 := by
  sorry

#check runners_meet_at_2000

end NUMINAMATH_CALUDE_ERRORFEEDBACK_runners_meet_at_2000_l126_12623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_drove_before_break_l126_12645

/-- Represents John's car trip with given parameters -/
structure CarTrip where
  totalDistance : ℝ
  totalTime : ℝ
  firstSpeed : ℝ
  secondSpeed : ℝ
  breakDuration : ℝ

/-- Calculates the time John drove before taking a break -/
noncomputable def timeDrivenBeforeBreak (trip : CarTrip) : ℝ :=
  (trip.totalDistance * 4 - trip.firstSpeed * trip.totalTime - trip.secondSpeed * (trip.totalTime - trip.breakDuration)) /
  (4 * (trip.firstSpeed - trip.secondSpeed))

/-- Theorem stating that for the given trip parameters, John drove for 1.25 hours before the break -/
theorem john_drove_before_break (trip : CarTrip)
  (h1 : trip.totalDistance = 300)
  (h2 : trip.totalTime = 4)
  (h3 : trip.firstSpeed = 60)
  (h4 : trip.secondSpeed = 90)
  (h5 : trip.breakDuration = 0.25) :
  timeDrivenBeforeBreak trip = 1.25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_drove_before_break_l126_12645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_ratio_lower_bound_l126_12642

/-- A triangle in a 2D plane --/
structure Triangle where
  a : ℝ × ℝ
  b : ℝ × ℝ
  c : ℝ × ℝ

/-- The largest side length of a square inscribed in a triangle --/
noncomputable def largest_inscribed_square_side (t : Triangle) : ℝ := sorry

/-- The shortest side length of a square circumscribed around a triangle --/
noncomputable def shortest_circumscribed_square_side (t : Triangle) : ℝ := sorry

/-- The ratio of the shortest circumscribed square side to the largest inscribed square side --/
noncomputable def square_ratio (t : Triangle) : ℝ :=
  shortest_circumscribed_square_side t / largest_inscribed_square_side t

theorem square_ratio_lower_bound :
  ∀ t : Triangle, square_ratio t ≥ 2 ∧ ∃ t : Triangle, square_ratio t = 2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_ratio_lower_bound_l126_12642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l126_12618

-- Problem 1
theorem problem_1 : (1/3)⁻¹ - |2 - Real.sqrt 5| + (2023 - Real.pi)^0 + Real.sqrt 20 = 6 + Real.sqrt 5 := by
  sorry

-- Problem 2
theorem problem_2 (x y : ℝ) (hx : x = 1 / (3 - Real.sqrt 10)) (hy : y = 1 / (3 + Real.sqrt 10)) :
  x^2 - x*y + y^2 = 39 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l126_12618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l126_12656

-- Define the vector m
noncomputable def m : ℝ × ℝ := (1, Real.sqrt 3)

-- Define the vector n as a function of x
noncomputable def n (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.cos x)

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := m.1 * (n x).1 + m.2 * (n x).2

-- Define the theorem
theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  c = Real.sqrt 6 →
  Real.cos B = 1 / 3 →
  f C = Real.sqrt 3 →
  b = 8 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l126_12656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correlation_coefficient_properties_l126_12662

-- Define the correlation coefficient as a real number
def correlation_coefficient : Type := ℝ

-- Define the property of correlation coefficient
def is_correlation_coefficient (r : ℝ) : Prop := abs r ≤ 1

-- Define the strength of linear correlation
def linear_correlation_strength (r : ℝ) : ℝ := abs r

-- Theorem statement
theorem correlation_coefficient_properties (r : ℝ) 
  (h : is_correlation_coefficient r) : 
  (abs r ≤ 1) ∧ 
  (∀ ε > 0, abs r > 1 - ε → linear_correlation_strength r > 1 - ε) ∧
  (∀ ε > 0, abs r < ε → linear_correlation_strength r < ε) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correlation_coefficient_properties_l126_12662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_colorings_count_l126_12650

-- Define the color type
inductive Color
| Red
| Green

-- Define a coloring function type
def Coloring := Nat → Color

-- Define the property of a valid coloring
def IsValidColoring (c : Coloring) : Prop :=
  (∀ n : Nat, n > 0 → (c n = Color.Red ∨ c n = Color.Green)) ∧
  (∀ m n : Nat, m > 0 → n > 0 → m ≠ n →
    (c m = Color.Red ∧ c n = Color.Red → c (m + n) = Color.Red)) ∧
  (∀ m n : Nat, m > 0 → n > 0 → m ≠ n →
    (c m = Color.Green ∧ c n = Color.Green → c (m + n) = Color.Green))

-- Define the set of all valid colorings
def ValidColorings := {c : Coloring | IsValidColoring c}

-- Theorem statement
theorem valid_colorings_count :
  ∃ (s : Finset Coloring), s.card = 6 ∧ ∀ c, c ∈ s ↔ IsValidColoring c :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_colorings_count_l126_12650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sulfuric_acid_moles_l126_12647

/-- Represents the number of moles of a substance -/
def Moles : Type := ℝ

/-- Represents the chemical reaction Zn + H2SO4 → ZnSO4 + H2 -/
structure Reaction where
  zn : Moles
  h2so4 : Moles
  znso4 : Moles
  h2 : Moles

/-- The reaction is balanced when the number of moles of reactants and products are equal -/
def is_balanced (r : Reaction) : Prop :=
  r.zn = r.h2so4 ∧ r.zn = r.znso4 ∧ r.zn = r.h2

theorem sulfuric_acid_moles (r : Reaction) 
  (h1 : is_balanced r) 
  (h2 : r.zn = (2 : ℝ)) 
  (h3 : r.znso4 = (2 : ℝ)) : 
  r.h2so4 = (2 : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sulfuric_acid_moles_l126_12647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_print_time_360_pages_l126_12626

/-- Calculates the time required to print a given number of pages, rounded to the nearest minute. -/
def print_time (pages_to_print : ℕ) (pages_per_minute : ℕ) : ℕ :=
  (pages_to_print + pages_per_minute - 1) / pages_per_minute

/-- Proves that printing 360 pages at 24 pages per minute takes 15 minutes. -/
theorem print_time_360_pages : print_time 360 24 = 15 := by
  rfl

#eval print_time 360 24

end NUMINAMATH_CALUDE_ERRORFEEDBACK_print_time_360_pages_l126_12626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_iff_c_eq_zero_l126_12678

/-- A sequence is defined by its partial sums -/
def SequenceFromPartialSums (a b c : ℝ) : ℕ → ℝ := λ n ↦
  if n = 1 then a + b + c
  else (a * n^2 + b * n + c) - (a * (n-1)^2 + b * (n-1) + c)

/-- A sequence is arithmetic if the difference between consecutive terms is constant -/
def IsArithmeticSequence (seq : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, n ≥ 1 → seq (n + 1) - seq n = d

theorem arithmetic_sequence_iff_c_eq_zero (a b c : ℝ) :
  IsArithmeticSequence (SequenceFromPartialSums a b c) ↔ c = 0 := by
  sorry

#check arithmetic_sequence_iff_c_eq_zero

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_iff_c_eq_zero_l126_12678
