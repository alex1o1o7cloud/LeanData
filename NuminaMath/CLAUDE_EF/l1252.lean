import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_silver_price_per_ounce_l1252_125286

/-- The price of an ounce of silver given a cube with specific dimensions and selling conditions -/
theorem silver_price_per_ounce 
  (cube_side : ℝ) 
  (weight_per_cubic_inch : ℝ) 
  (selling_percentage : ℝ) 
  (selling_price : ℝ) 
  (h1 : cube_side = 3)
  (h2 : weight_per_cubic_inch = 6)
  (h3 : selling_percentage = 1.1)
  (h4 : selling_price = 4455) : 
  (selling_price / selling_percentage) / (cube_side^3 * weight_per_cubic_inch) = 25 := by
  sorry

#check silver_price_per_ounce

end NUMINAMATH_CALUDE_ERRORFEEDBACK_silver_price_per_ounce_l1252_125286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_proof_l1252_125212

/-- Represents the rate of interest per annum as a percentage -/
noncomputable def rate : ℚ := 8

/-- Simple interest calculation function -/
def simple_interest (principal time rate : ℚ) : ℚ :=
  principal * rate * time / 100

theorem interest_rate_proof :
  let loan_b : ℚ := 5000
  let time_b : ℚ := 2
  let loan_c : ℚ := 3000
  let time_c : ℚ := 4
  let total_interest : ℚ := 1760
  simple_interest loan_b time_b rate + simple_interest loan_c time_c rate = total_interest :=
by
  -- Unfold the definitions
  unfold simple_interest rate
  -- Perform algebraic simplifications
  simp [add_div, mul_div_assoc, mul_comm, mul_assoc]
  -- Check equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_proof_l1252_125212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_short_covering_sequence_property_l1252_125266

def isCoveringSequence (d : List ℕ) : Prop :=
  ∃ c : List ℕ, ∀ n : ℕ, ∃ i k : ℕ, i < d.length ∧ n = c.get! i + k * d.get! i

def isShortCoveringSequence (d : List ℕ) : Prop :=
  isCoveringSequence d ∧ ∀ i : ℕ, i < d.length → ¬isCoveringSequence (d.removeNth i)

theorem short_covering_sequence_property
  (d : List ℕ) (a : List ℕ) (p : ℕ) (k : ℕ) :
  isShortCoveringSequence d →
  (∀ n : ℕ, ∃ i j : ℕ, i < d.length ∧ n = a.get! i + j * d.get! i) →
  Nat.Prime p →
  (∀ i : ℕ, i < k → p ∣ d.get! i) →
  (∀ i : ℕ, k ≤ i ∧ i < d.length → ¬(p ∣ d.get! i)) →
  (Finset.range p).val = (Finset.image (λ i => a.get! i % p) (Finset.range k)).val :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_short_covering_sequence_property_l1252_125266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_D_determines_location_l1252_125221

-- Define a type for locations
structure Location where
  longitude : ℝ
  latitude : ℝ

-- Define a function to check if a description can determine a unique location
def canDetermineLocation (description : String) : Prop :=
  ∃! (loc : Location), true  -- We replace 'descriptionMatchesLocation' with a trivial condition

-- Define the four options
def optionA : String := "Row 2 of Huayu Cinema"
def optionB : String := "Central Street of Zhaoyuan County"
def optionC : String := "Northward 30 degrees east"
def optionD : String := "East longitude 118 degrees, north latitude 40 degrees"

-- Theorem stating that only option D can determine a unique location
theorem only_D_determines_location :
  (¬ canDetermineLocation optionA) ∧
  (¬ canDetermineLocation optionB) ∧
  (¬ canDetermineLocation optionC) ∧
  (canDetermineLocation optionD) := by
  sorry

-- You can add more definitions or lemmas here if needed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_D_determines_location_l1252_125221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_difference_l1252_125292

-- Define the circles
def C1 (x y : ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = 1
def C2 (x y : ℝ) : Prop := (x - 4)^2 + (y - 5)^2 = 9

-- Define the points
def M (x y : ℝ) : Prop := C1 x y
def N (x y : ℝ) : Prop := C2 x y
def P (x : ℝ) : Prop := True  -- P is any point on the x-axis

-- Define the distance function
noncomputable def dist (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- State the theorem
theorem max_distance_difference :
  ∃ (xP : ℝ), ∀ (xM yM xN yN : ℝ),
    M xM yM → N xN yN → P xP →
    dist xP 0 xN yN - dist xP 0 xM yM ≤ 9 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_difference_l1252_125292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_area_example_l1252_125224

/-- The area of an isosceles trapezoid with given dimensions -/
noncomputable def isosceles_trapezoid_area (leg : ℝ) (base1 : ℝ) (base2 : ℝ) : ℝ :=
  let height := Real.sqrt (leg^2 - ((base2 - base1) / 2)^2)
  (base1 + base2) * height / 2

/-- Theorem: The area of an isosceles trapezoid with legs of length 5 and bases of length 9 and 15 is 48 -/
theorem isosceles_trapezoid_area_example : isosceles_trapezoid_area 5 9 15 = 48 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_area_example_l1252_125224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l1252_125295

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sqrt x + Real.sqrt (8 - x)

-- State the theorem
theorem f_minimum_value :
  ∃ (x : ℝ), x ∈ Set.Icc 0 8 ∧ 
  (∀ (y : ℝ), y ∈ Set.Icc 0 8 → f y ≥ f x) ∧
  f x = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l1252_125295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_yard_conversion_l1252_125247

-- Define the conversion factors
def yard_to_feet : ℝ := 3
def foot_to_meter : ℝ := 0.3048

-- Define the volume conversion function
def cubic_yard_to_cubic_meter : ℝ := 
  (yard_to_feet * foot_to_meter) ^ 3

-- Theorem statement
theorem cubic_yard_conversion :
  abs (cubic_yard_to_cubic_meter - 0.764554) < 0.000001 := by
  -- Proof steps would go here
  sorry

#eval cubic_yard_to_cubic_meter

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_yard_conversion_l1252_125247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_simplification_l1252_125267

theorem factorial_simplification : 
  (12 * 11 * 10 * Nat.factorial 9) / (10 * Nat.factorial 9 + 2 * Nat.factorial 9) = 110 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_simplification_l1252_125267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_intersection_l1252_125258

/-- An ellipse C with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- A line L with slope k -/
structure Line where
  k : ℝ
  h_k : k > -2

/-- The problem statement -/
theorem ellipse_and_line_intersection (C : Ellipse) (L : Line) (t : ℝ) :
  C.a^2 = 4 ∧ C.b^2 = 2 ∧ 4 - 3*Real.sqrt 2 / 4 ≤ t ∧ t < 5 :=
by
  have h1 : C.a^2 - C.b^2 = 2 := sorry
  have h2 : 2 / C.a^2 + 1 / C.b^2 = 1 := sorry
  have h3 : ∃ (A B : ℝ × ℝ), A ≠ B ∧ 
    (A.1^2 / C.a^2 + A.2^2 / C.b^2 = 1) ∧
    (B.1^2 / C.a^2 + B.2^2 / C.b^2 = 1) ∧
    (A.2 = L.k * (A.1 + 1)) ∧ 
    (B.2 = L.k * (B.1 + 1)) := sorry
  have h4 : ∀ (A B : ℝ × ℝ), A ≠ B →
    let M := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
    |2 * M.1 + M.2 + t| / Real.sqrt 5 = 3 * Real.sqrt 5 / 5 := sorry
  have h5 : t > 2 := sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_intersection_l1252_125258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_identification_l1252_125227

-- Define what a power function is
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α

-- Define the given functions
noncomputable def f1 (x : ℝ) : ℝ := x^2 + 1
noncomputable def f2 (x : ℝ) : ℝ := 2^x
noncomputable def f3 (x : ℝ) : ℝ := 1 / (x^2)
noncomputable def f4 (x : ℝ) : ℝ := (x - 1)^2
noncomputable def f5 (x : ℝ) : ℝ := x^5
noncomputable def f6 (x : ℝ) : ℝ := x^(x + 1)

-- State the theorem
theorem power_function_identification :
  (¬ isPowerFunction f1) ∧
  (¬ isPowerFunction f2) ∧
  (isPowerFunction f3) ∧
  (¬ isPowerFunction f4) ∧
  (isPowerFunction f5) ∧
  (¬ isPowerFunction f6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_identification_l1252_125227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1252_125237

-- Define the ⊕ operation
noncomputable def circleplus (a b : ℝ) : ℝ :=
  if a ≥ b then a else b^2

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (circleplus 1 x) + (circleplus 2 x)

-- Theorem statement
theorem max_value_of_f :
  ∃ (m : ℝ), m = 18 ∧ ∀ x, x ∈ Set.Icc (-2 : ℝ) 3 → f x ≤ m :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1252_125237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l1252_125298

noncomputable def g (x : ℝ) : ℝ := (7^x - 1) / (7^x + 1)

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x := by
  intro x
  simp [g]
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l1252_125298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_establishment_charges_upper_bound_l1252_125208

/-- Represents the establishment charges of a techno company --/
def establishment_charges : ℝ := sorry

/-- The total number of machines in the factory --/
def total_machines : ℝ := 14

/-- The annual manufacturing costs when all machines are operational --/
def manufacturing_costs : ℝ := 42000

/-- The annual output when all machines are operational --/
def annual_output : ℝ := 70000

/-- The profit percentage for shareholders --/
def profit_percentage : ℝ := 0.125

/-- The number of machines that remain closed --/
def closed_machines : ℝ := 7.14

/-- The percentage decrease in profit when machines are closed --/
def profit_decrease_percentage : ℝ := 0.125

/-- Theorem stating that the establishment charges are less than or equal to 21376.25 --/
theorem establishment_charges_upper_bound :
  establishment_charges ≤ 21376.25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_establishment_charges_upper_bound_l1252_125208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_has_four_real_solutions_l1252_125257

theorem equation_has_four_real_solutions :
  ∃ (s : Finset ℝ), (∀ x ∈ s, (6 * x) / (x^2 + 2 * x + 5) + (4 * x) / (x^2 - 4 * x + 5) = -2/3) ∧ 
                    Finset.card s = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_has_four_real_solutions_l1252_125257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_pi_sixth_eq_one_b_plus_c_range_l1252_125213

open Real

noncomputable def f (x : ℝ) : ℝ := 
  sin (Real.pi/4 + x) * sin (Real.pi/4 - x) + Real.sqrt 3 * sin x * cos x

-- Part 1: Prove f(π/6) = 1
theorem f_pi_sixth_eq_one : f (Real.pi/6) = 1 := by sorry

-- Part 2: Prove the range of b + c
theorem b_plus_c_range (A B C a b c : ℝ) :
  0 < A → A < Real.pi/2 →  -- Acute triangle condition
  0 < B → B < Real.pi/2 →
  0 < C → C < Real.pi/2 →
  A + B + C = Real.pi →    -- Triangle angle sum
  f (A/2) = 1 →      -- Given condition
  a = 2 →            -- Given condition
  a * sin B = b * sin A →  -- Law of sines
  a * sin C = c * sin A →  -- Law of sines
  2 * Real.sqrt 3 < b + c ∧ b + c ≤ 4 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_pi_sixth_eq_one_b_plus_c_range_l1252_125213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alloy_composition_l1252_125293

/-- Represents the ratio of a metal in an alloy -/
structure MetalRatio where
  gold : ℝ
  silver : ℝ
  copper : ℝ

/-- The density of the alloy relative to water -/
noncomputable def alloyDensity (r : MetalRatio) : ℝ :=
  (10 * r.gold + 4 * r.silver + 6 * r.copper) / (r.gold + r.silver + r.copper)

/-- The fraction of silver in the alloy -/
noncomputable def silverFraction (r : MetalRatio) : ℝ :=
  r.silver / (r.gold + r.silver + r.copper)

/-- The theorem stating the unique solution for the alloy composition -/
theorem alloy_composition :
  ∀ r : MetalRatio,
    r.gold > 0 ∧ r.silver > 0 ∧
    alloyDensity r = 8 ∧
    silverFraction r ≥ 1/3 →
    r.gold = 2 * r.silver ∧ r.copper = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alloy_composition_l1252_125293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l1252_125241

/-- Given vectors a and b, if (a - b) is parallel to (2a + b), then λ = 2/3 -/
theorem parallel_vectors_lambda (a b : ℝ × ℝ) (l : ℝ) :
  a = (2, 3) →
  b = (l, 1) →
  (∃ k : ℝ, a - b = k • (2 • a + b)) →
  l = 2/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l1252_125241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_q_value_l1252_125281

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the area of a triangle given its three vertices -/
noncomputable def triangleArea (a b c : Point) : ℝ :=
  (1/2) * abs (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y))

theorem triangle_q_value (q : ℝ) :
  let a : Point := ⟨3, 15⟩
  let b : Point := ⟨15, 0⟩
  let c : Point := ⟨0, q⟩
  triangleArea a b c = 36 →
  q = 12.75 := by
  sorry

#check triangle_q_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_q_value_l1252_125281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_eq_one_sufficient_not_necessary_l1252_125223

/-- Two lines are perpendicular if their slopes multiply to -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- The slope of line l1: ax + (a-1)y - 1 = 0 -/
noncomputable def slope_l1 (a : ℝ) : ℝ := -a / (a - 1)

/-- The slope of line l2: (a-1)x + (2a+3)y - 3 = 0 -/
noncomputable def slope_l2 (a : ℝ) : ℝ := -(a - 1) / (2*a + 3)

/-- The statement that a=1 is a sufficient but not necessary condition for the perpendicularity of l1 and l2 -/
theorem a_eq_one_sufficient_not_necessary :
  (∀ a : ℝ, a = 1 → perpendicular (slope_l1 a) (slope_l2 a)) ∧
  ¬(∀ a : ℝ, perpendicular (slope_l1 a) (slope_l2 a) → a = 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_eq_one_sufficient_not_necessary_l1252_125223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_M_l1252_125248

/-- Represents a four-digit positive integer in base 10 -/
def FourDigitInt : Type := { n : ℕ // 1000 ≤ n ∧ n < 10000 }

/-- Converts a natural number to its base-4 representation -/
def toBase4 (n : ℕ) : ℕ := sorry

/-- Converts a natural number to its base-7 representation -/
def toBase7 (n : ℕ) : ℕ := sorry

/-- Computes T as the sum of base-4 and base-7 representations of M treated as base-10 integers -/
def computeT (M : FourDigitInt) : ℕ := toBase4 M.val + toBase7 M.val

/-- Checks if the two rightmost digits of two numbers are the same -/
def sameLastTwoDigits (a b : ℕ) : Prop := a % 100 = b % 100

/-- Assume FourDigitInt is finite -/
instance : Fintype FourDigitInt := sorry

/-- Assume sameLastTwoDigits is decidable -/
instance (a b : ℕ) : Decidable (sameLastTwoDigits a b) := sorry

/-- The main theorem stating that there are 32 valid choices for M -/
theorem count_valid_M : 
  (Finset.filter (fun M : FourDigitInt => sameLastTwoDigits (computeT M) (3 * M.val)) 
    Finset.univ).card = 32 := by sorry

#check count_valid_M

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_M_l1252_125248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_n_l1252_125235

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

noncomputable def series_sum (n : ℕ+) : ℝ :=
  ∑' k : ℕ, (floor ((n : ℝ) + 2^k) / 2^(k+1))

theorem series_sum_equals_n (n : ℕ+) : ⌊series_sum n⌋ = n.val := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_n_l1252_125235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sam_distance_l1252_125270

/-- Calculates the distance traveled given rate and time -/
noncomputable def distance (rate : ℝ) (time : ℝ) : ℝ := rate * time

/-- Calculates the rate given distance and time -/
noncomputable def rate (distance : ℝ) (time : ℝ) : ℝ := distance / time

theorem sam_distance (marguerite_distance marguerite_time sam_time : ℝ) 
  (h1 : marguerite_distance = 150)
  (h2 : marguerite_time = 3)
  (h3 : sam_time = 4) :
  distance (rate marguerite_distance marguerite_time) sam_time = 200 := by
  sorry

#check sam_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sam_distance_l1252_125270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fish_speed_theorem_l1252_125229

/-- Calculates the speed of a fish in still water given its upstream and downstream speeds -/
noncomputable def fish_speed_in_still_water (upstream_speed downstream_speed : ℝ) : ℝ :=
  (upstream_speed + downstream_speed) / 2

/-- Theorem: The speed of a fish in still water is 45 kmph given its upstream speed is 35 kmph and downstream speed is 55 kmph -/
theorem fish_speed_theorem (upstream_speed downstream_speed : ℝ)
  (h1 : upstream_speed = 35)
  (h2 : downstream_speed = 55) :
  fish_speed_in_still_water upstream_speed downstream_speed = 45 := by
  sorry

-- Use #eval only with nat or other computable types
#eval (35 : ℕ) + (55 : ℕ)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fish_speed_theorem_l1252_125229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_poultry_farm_count_l1252_125214

def total_poultry (num_hens num_ducks num_geese : ℕ) 
  (male_female_ratio : ℚ) 
  (chicks_per_hen ducklings_per_duck goslings_per_goose : ℕ) : ℕ :=
  let num_female_hens := (↑num_hens * male_female_ratio.den / (male_female_ratio.num + male_female_ratio.den)).floor.toNat
  let num_female_ducks := (↑num_ducks * male_female_ratio.den / (male_female_ratio.num + male_female_ratio.den)).floor.toNat
  let num_female_geese := (↑num_geese * male_female_ratio.den / (male_female_ratio.num + male_female_ratio.den)).floor.toNat
  num_hens + num_ducks + num_geese + 
  num_female_hens * chicks_per_hen + 
  num_female_ducks * ducklings_per_duck + 
  num_female_geese * goslings_per_goose

theorem poultry_farm_count : 
  total_poultry 25 10 5 (1/4 : ℚ) 6 8 3 = 236 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_poultry_farm_count_l1252_125214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_on_interval_l1252_125239

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log (1/2)

-- Define the inverse function g
noncomputable def g (x : ℝ) : ℝ := (1/2) ^ x

-- State the theorem
theorem range_of_g_on_interval :
  let a := 1
  let b := 2
  let lower_bound := (1/2)^b
  let upper_bound := (1/2)^a
  (∀ x, f (g x) = x) →
  (∀ x, g (f x) = x) →
  (∀ x y, a ≤ x → x < y → y ≤ b → g y < g x) →
  {y | ∃ x, a ≤ x ∧ x ≤ b ∧ y = g x} = Set.Icc lower_bound upper_bound :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_on_interval_l1252_125239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_to_long_base_approx_l1252_125269

/-- An isosceles trapezoid with given dimensions -/
structure IsoscelesTrapezoid where
  longBase : ℝ
  shortBase : ℝ
  side : ℝ
  baseAngle : ℝ

/-- Calculate the height of the trapezoid -/
noncomputable def trapezoidHeight (t : IsoscelesTrapezoid) : ℝ :=
  t.side / 2 * Real.sin t.baseAngle

/-- Calculate the total area of the trapezoid -/
noncomputable def totalArea (t : IsoscelesTrapezoid) : ℝ :=
  (t.longBase + t.shortBase) / 2 * trapezoidHeight t

/-- Calculate the area of the upper part of the trapezoid -/
noncomputable def upperArea (t : IsoscelesTrapezoid) : ℝ :=
  (t.longBase + t.shortBase / 2) / 2 * trapezoidHeight t

/-- The fraction of the area closer to the longer base -/
noncomputable def fractionToLongBase (t : IsoscelesTrapezoid) : ℝ :=
  upperArea t / totalArea t

/-- Theorem stating that the fraction of the area closer to the longer base
    is approximately 0.8 for the given trapezoid dimensions -/
theorem fraction_to_long_base_approx (t : IsoscelesTrapezoid)
    (h1 : t.longBase = 150)
    (h2 : t.shortBase = 100)
    (h3 : t.side = 130)
    (h4 : t.baseAngle = 75 * π / 180) :
    ∃ ε > 0, |fractionToLongBase t - 0.8| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_to_long_base_approx_l1252_125269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_even_n_for_trig_inequality_l1252_125217

theorem largest_even_n_for_trig_inequality :
  ∃ (n : ℕ), 
    (Even n ∧ n > 0) ∧
    (∀ (x : ℝ), Real.sin x ^ n - Real.cos x ^ n ≤ 1) ∧
    (∀ (m : ℕ), m > n → Even m → ∃ (y : ℝ), Real.sin y ^ m - Real.cos y ^ m > 1) ∧
    n = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_even_n_for_trig_inequality_l1252_125217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_tangent_theorem_l1252_125203

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the point P
structure Point (α : Type) where
  x : α
  y : α

-- Define the condition for P
def valid_point (P : Point ℝ) : Prop :=
  P.y ≠ 0 ∧ ∃ (A B : Point ℝ), 
    parabola A.x A.y ∧ 
    parabola B.x B.y ∧ 
    (B.y - A.y) * (P.x - A.x) = -(B.x - A.x) * (P.y - A.y)

-- Define R as the intersection of l_P and x-axis
def R : Point ℝ := ⟨2, 0⟩

-- Define the ratio |PQ|/|QR|
noncomputable def ratio (P Q R : Point ℝ) : ℝ :=
  Real.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2) / 
  Real.sqrt ((Q.x - R.x)^2 + (Q.y - R.y)^2)

-- State the theorem
theorem parabola_tangent_theorem (P : Point ℝ) (hP : valid_point P) :
  (∃ Q : Point ℝ, ratio P Q R = Real.sqrt 8) ∧
  (∀ Q : Point ℝ, ratio P Q R ≥ Real.sqrt 8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_tangent_theorem_l1252_125203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_values_l1252_125240

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 4 * x + 7 else 10 - 3 * x

-- Theorem to prove f(-2) = -1 and f(3) = 1
theorem f_values : f (-2) = -1 ∧ f 3 = 1 := by
  constructor
  · -- Prove f(-2) = -1
    simp [f]
    norm_num
  · -- Prove f(3) = 1
    simp [f]
    norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_values_l1252_125240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_theorem_l1252_125275

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x + 2

-- Define the inverse function f_inv
noncomputable def f_inv (x : ℝ) : ℝ := 1 - Real.sqrt (x - 1)

-- State the theorem
theorem inverse_function_theorem (x : ℝ) (h1 : x > 1) :
  f (f_inv x) = x ∧ f_inv (f x) = x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_theorem_l1252_125275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_road_trip_water_bottles_l1252_125218

def water_bottles_needed 
  (family_size : ℕ) 
  (water_per_person_per_hour : ℚ) 
  (trip_duration : ℕ) : ℕ :=
  let total_water := family_size * water_per_person_per_hour * trip_duration
  (total_water.ceil.toNat)

theorem road_trip_water_bottles :
  water_bottles_needed 4 (1/2) 16 = 32 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_road_trip_water_bottles_l1252_125218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_formula_is_correct_l1252_125211

/-- The area of a convex quadrilateral with side lengths a, b, c, d and diagonal lengths e, f. -/
noncomputable def quadrilateral_area (a b c d e f : ℝ) : ℝ :=
  (1/4) * Real.sqrt (4 * e^2 * f^2 - (a^2 + c^2 - b^2 - d^2)^2)

/-- Predicate to check if a quadrilateral is convex given its side lengths and diagonal lengths. -/
def IsConvexQuadrilateral (a b c d e f : ℝ) : Prop :=
  sorry

/-- Predicate to check if a given value is the area of a quadrilateral with given side and diagonal lengths. -/
def IsAreaOfQuadrilateral (S a b c d e f : ℝ) : Prop :=
  sorry

/-- Theorem stating that the given formula correctly computes the area of a convex quadrilateral. -/
theorem quadrilateral_area_formula_is_correct
  (a b c d e f : ℝ)
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (he : e > 0) (hf : f > 0)
  (hconvex : IsConvexQuadrilateral a b c d e f) :
  ∃ (S : ℝ), S = quadrilateral_area a b c d e f ∧ IsAreaOfQuadrilateral S a b c d e f :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_formula_is_correct_l1252_125211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l1252_125268

theorem trig_identity (α : ℝ) : 
  (Real.cos (8 * α) * Real.tan (4 * α) - Real.sin (8 * α)) * 
  (Real.cos (8 * α) * (1 / Real.tan (4 * α)) + Real.sin (8 * α)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l1252_125268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_property_l1252_125272

noncomputable def geometric_sum (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

theorem geometric_sequence_property (a₁ : ℝ) (q : ℝ) :
  (∃ S₃ S₆ S₉ : ℝ,
    S₃ = geometric_sum a₁ q 3 ∧
    S₆ = geometric_sum a₁ q 6 ∧
    S₉ = geometric_sum a₁ q 9 ∧
    2 * S₆ = S₃ + S₉) →
  q^3 = -1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_property_l1252_125272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l1252_125262

noncomputable def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

noncomputable def g (x : ℝ) : ℝ := f (x - 1) + 1

-- Theorem: g is an odd function
theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x := by
  intro x
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l1252_125262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_term_is_one_tenth_l1252_125283

/-- A sequence satisfying the given recurrence relation -/
def mySequence (a : ℕ+ → ℚ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ+, a (n + 1) = a n / (1 + a n)

/-- The 10th term of the sequence is 1/10 -/
theorem tenth_term_is_one_tenth (a : ℕ+ → ℚ) (h : mySequence a) : a 10 = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_term_is_one_tenth_l1252_125283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_of_A_l1252_125230

/-- The speed of person A in meters per minute -/
def a : ℕ := sorry

/-- The speed of person B in meters per minute -/
def b : ℕ := sorry

/-- The time it takes for A and B to meet at point C -/
noncomputable def τ : ℚ := sorry

/-- The distance between points A and B -/
noncomputable def D : ℚ := sorry

/-- a is greater than b -/
axiom h1 : a > b

/-- b is not a factor of a -/
axiom h2 : ¬(a % b = 0)

/-- The distance covered by A and B when they meet at C -/
axiom h3 : a * τ + b * τ = D

/-- The new meeting point when A starts 2 minutes earlier -/
axiom h4 : a * (τ + 2) + b * τ = D + 42

/-- The speed of person A is 21 meters per minute -/
theorem speed_of_A : a = 21 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_of_A_l1252_125230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_can_eat_all_jam_l1252_125225

/-- Represents the state of jars and the eating strategy -/
structure JamEatingProblem where
  n : ℕ  -- number of jars
  k : ℕ  -- number of jars chosen each day
  max_fraction : ℚ  -- maximum fraction of total jam in any jar
  jam_amounts : Fin n → ℚ  -- amount of jam in each jar

/-- The conditions of the problem -/
def valid_problem (p : JamEatingProblem) : Prop :=
  p.n = 1000 ∧
  p.k = 100 ∧
  p.max_fraction = 1 / 100 ∧
  ∀ i, 0 ≤ p.jam_amounts i ∧ p.jam_amounts i ≤ p.max_fraction

/-- Placeholder for the strategy that empties all jars -/
def strategy_empties_all_jars (p : JamEatingProblem) (strategy : ℕ → Fin p.n → Bool) : Prop :=
  sorry

/-- The theorem to be proved -/
theorem can_eat_all_jam (p : JamEatingProblem) (h : valid_problem p) :
  ∃ strategy, strategy_empties_all_jars p strategy :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_can_eat_all_jam_l1252_125225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_subset_with_unique_sum_representation_l1252_125263

-- Define the set M as the rational numbers in the open interval (0,1)
def M : Set ℚ := {q : ℚ | 0 < q ∧ q < 1}

-- State the theorem
theorem exists_subset_with_unique_sum_representation :
  ∃ (A : Set ℚ), A ⊆ M ∧
  ∀ (q : ℚ), q ∈ M →
    ∃! (S : Finset ℚ), S.toSet ⊆ A ∧ q = S.sum id ∧ S.Nonempty :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_subset_with_unique_sum_representation_l1252_125263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_perpendicular_to_parallel_plane_l1252_125277

/-- A type representing planes in 3D space -/
structure Plane where

/-- A type representing lines in 3D space -/
structure Line where

/-- Perpendicularity relation between a line and a plane -/
def perpendicular (l : Line) (p : Plane) : Prop := sorry

/-- Parallelism relation between two planes -/
def parallel (p1 p2 : Plane) : Prop := sorry

theorem line_perpendicular_to_parallel_plane 
  (α β : Plane) (l : Line) 
  (h1 : α ≠ β) 
  (h2 : perpendicular l α) 
  (h3 : parallel α β) : 
  perpendicular l β := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_perpendicular_to_parallel_plane_l1252_125277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_geq_3_range_of_a_l1252_125284

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 - x + 3 else x + 2/x

-- Theorem for part (I)
theorem solution_set_f_geq_3 :
  {x : ℝ | f x ≥ 3} = {x : ℝ | x ≤ 0 ∨ x = 1 ∨ x ≥ 2} := by sorry

-- Theorem for part (II)
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f x ≥ |x/2 + a|) ↔ -47/16 ≤ a ∧ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_geq_3_range_of_a_l1252_125284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_theorem_l1252_125271

open Set

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]
variable (A : Set E)
variable (f : A → A)
variable (c : ℝ)

/-- The Lipschitz condition for f with constant c -/
def isLipschitz (f : A → A) (c : ℝ) : Prop :=
  ∀ x y : A, ‖(f x : E) - (f y : E)‖ ≤ c * ‖(x : E) - (y : E)‖

theorem fixed_point_theorem (hA : IsClosed A) (hAne : Nonempty A) (hc : c < 1) 
    (hf : isLipschitz A f c) : 
  ∃! x : A, f x = x :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_theorem_l1252_125271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_l1252_125291

/-- Circle represented by its equation in the form x^2 + y^2 + ax + by + c = 0 -/
structure Circle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculate the center and radius of a circle -/
noncomputable def circle_center_radius (c : Circle) : (ℝ × ℝ) × ℝ :=
  let center_x := -c.a / 2
  let center_y := -c.b / 2
  let radius := Real.sqrt ((c.a^2 + c.b^2) / 4 - c.c)
  ((center_x, center_y), radius)

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Check if two circles are intersecting -/
noncomputable def are_intersecting (c1 c2 : Circle) : Prop :=
  let ((x1, y1), r1) := circle_center_radius c1
  let ((x2, y2), r2) := circle_center_radius c2
  let d := distance (x1, y1) (x2, y2)
  abs (r1 - r2) < d ∧ d < r1 + r2

theorem circles_intersect : 
  let c1 : Circle := ⟨-4, 2, 4⟩
  let c2 : Circle := ⟨-2, 6, 6⟩
  are_intersecting c1 c2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_l1252_125291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_increasing_interval_l1252_125259

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (a - 1) * x^2 + a * x + 3

-- State the theorem
theorem even_function_increasing_interval (a : ℝ) :
  (∀ x, f a x = f a (-x)) →  -- f is an even function
  {x : ℝ | ∀ y, y < x → f a y < f a x} = Set.Iic 0 := by
  sorry

-- Note: Set.Iic 0 represents the set (-∞, 0] in Lean

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_increasing_interval_l1252_125259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_less_than_N_l1252_125273

-- Define the polynomials
noncomputable def p (x y : ℝ) : ℝ := (x^2 + x + 2*y)^5
noncomputable def q (x : ℝ) : ℝ := (3/x - x)^7

-- Define M as the coefficient of x^4y^2 in the expansion of p
noncomputable def M : ℝ := 120

-- Define N as the sum of coefficients in the expansion of q
noncomputable def N : ℝ := 128

-- Theorem statement
theorem M_less_than_N : M < N := by
  -- We replace the proof with sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_less_than_N_l1252_125273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_trig_expression_l1252_125264

theorem min_value_trig_expression (α : ℝ) (h : α ∈ Set.Ioo 0 (π / 2)) :
  (Real.sin α)^3 / (Real.cos α) + (Real.cos α)^3 / (Real.sin α) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_trig_expression_l1252_125264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_middle_joints_capacity_l1252_125226

/-- Represents the capacity of each joint in the bamboo tube -/
def bamboo_sequence : Fin 9 → ℝ := sorry

/-- The common difference of the arithmetic sequence -/
def common_difference : ℝ := sorry

/-- The sum of the first three terms is 4.5 -/
axiom lower_sum : bamboo_sequence 0 + bamboo_sequence 1 + bamboo_sequence 2 = 4.5

/-- The sum of the last four terms is 3.8 -/
axiom upper_sum : bamboo_sequence 5 + bamboo_sequence 6 + bamboo_sequence 7 + bamboo_sequence 8 = 3.8

/-- The sequence is an arithmetic progression -/
axiom is_arithmetic_seq : ∀ (i : Fin 8), bamboo_sequence (i.succ) - bamboo_sequence i = common_difference

/-- The theorem to be proved -/
theorem middle_joints_capacity :
  bamboo_sequence 3 + bamboo_sequence 4 = 2.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_middle_joints_capacity_l1252_125226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_polynomial_l1252_125232

/-- The polynomial x^5 + x^5 - 6x^4 - 4x^3 + 13x^2 + 5x - 10 -/
def f (x : ℂ) : ℂ := 2*x^5 - 6*x^4 - 4*x^3 + 13*x^2 + 5*x - 10

/-- The set of rational roots of the polynomial -/
def rational_roots : Set ℚ := {1, -2}

/-- The number of complex roots of the polynomial -/
def complex_roots_count : ℕ := 4

theorem roots_of_polynomial :
  (∀ q ∈ rational_roots, f (↑q) = 0) ∧ 
  (∃ z₁ z₂ z₃ z₄ : ℂ, f z₁ = 0 ∧ f z₂ = 0 ∧ f z₃ = 0 ∧ f z₄ = 0 ∧ 
    z₁ ≠ z₂ ∧ z₁ ≠ z₃ ∧ z₁ ≠ z₄ ∧ z₂ ≠ z₃ ∧ z₂ ≠ z₄ ∧ z₃ ≠ z₄) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_polynomial_l1252_125232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_in_hexagon_configuration_l1252_125261

/-- The side length of each hexagon -/
def hexagon_side_length : ℝ := 2

/-- The number of hexagons in the configuration -/
def num_hexagons : ℕ := 7

/-- Represents the configuration of hexagons -/
structure HexagonConfiguration where
  side_length : ℝ
  num_hexagons : ℕ
  is_centered : Bool

/-- The specific configuration in the problem -/
def problem_configuration : HexagonConfiguration :=
  { side_length := hexagon_side_length
  , num_hexagons := num_hexagons
  , is_centered := true }

/-- The area of the triangle formed by vertices closest to the center -/
noncomputable def triangle_area (config : HexagonConfiguration) : ℝ := 4 * Real.sqrt 3

/-- Theorem stating the area of the triangle in the given configuration -/
theorem triangle_area_in_hexagon_configuration :
  triangle_area problem_configuration = 4 * Real.sqrt 3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_in_hexagon_configuration_l1252_125261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1252_125255

noncomputable section

/-- The function f(x) = x^2 + px + q -/
def f (p q : ℝ) (x : ℝ) : ℝ := x^2 + p*x + q

/-- The function g(x) = x + 1/x^2 -/
def g (x : ℝ) : ℝ := x + 1/x^2

/-- The interval [1, 2] -/
def I : Set ℝ := Set.Icc 1 2

theorem max_value_of_f (p q : ℝ) :
  (∃ (x : ℝ), x ∈ I ∧ IsLocalMin (f p q) x ∧ IsLocalMin g x) →
  (∃ (y : ℝ), y ∈ I ∧ ∀ (z : ℝ), z ∈ I → f p q z ≤ f p q y) →
  (∃ (y : ℝ), y ∈ I ∧ f p q y = 4 - (5/2) * Real.rpow 2 (1/3) + Real.rpow 4 (1/3)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1252_125255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_f_at_one_l1252_125204

noncomputable def f (x : ℝ) : ℝ := (x^2 - 10*x + 9) / (x^2 - 9*x + 8)

theorem limit_of_f_at_one :
  (∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x - 1| ∧ |x - 1| < δ → |f x - 8/7| < ε) ∧
  f 1 = 0/0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_f_at_one_l1252_125204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_on_line_l1252_125222

theorem angle_on_line (α : ℝ) : 
  (∃ (x y : ℝ), y = -2 * x ∧ x * Real.cos α = y * Real.sin α) →  -- terminal side on y = -2x
  Real.sin α > 0 →                                               -- sin α > 0
  Real.cos α = -Real.sqrt 5 / 5 ∧ Real.tan α = -2 := by          -- conclusion
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_on_line_l1252_125222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_and_perpendicular_l1252_125276

noncomputable section

def i : ℝ × ℝ := (1, 0)
def j : ℝ × ℝ := (0, 1)

def OA : ℝ × ℝ := (1, 2)
def OB : ℝ × ℝ := (2, -4)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

noncomputable def cosine_angle (v w : ℝ × ℝ) : ℝ :=
  dot_product v w / (magnitude v * magnitude w)

def P : ℝ × ℝ := ((OA.1 + OB.1) / 2, (OA.2 + OB.2) / 2)

def OP : ℝ × ℝ := P

theorem vector_angle_and_perpendicular :
  (cosine_angle OA OB = -3/5) ∧
  (∃ k : ℝ, dot_product OP (OA.1 + k * OB.1, OA.2 + k * OB.2) = 0 ∧ k = 1/14) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_and_perpendicular_l1252_125276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1252_125233

-- Define the function f
noncomputable def f (c : ℝ) (x : ℝ) : ℝ := (c * x - 1) / (x + 1)

-- Define the function g
noncomputable def g (c : ℝ) (x : ℝ) : ℝ := f c (Real.exp x)

theorem problem_solution (c : ℝ) (h : f c 1 = 0) :
  c = 1 ∧
  (∀ x₁ x₂ : ℝ, 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 2 → f c x₁ < f c x₂) ∧
  (∀ x : ℝ, g c (-x) = -(g c x)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1252_125233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_z_values_l1252_125205

def is_valid_number (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def reverse_digits (n : ℕ) : ℕ :=
  let d1 := n / 1000
  let d2 := (n / 100) % 10
  let d3 := (n / 10) % 10
  let d4 := n % 10
  1000 * d4 + 100 * d3 + 10 * d2 + d1

theorem distinct_z_values (x y : ℕ) (h1 : is_valid_number x) (h2 : is_valid_number y) 
  (h3 : y = reverse_digits x) :
  ∃ (S : Finset ℕ), (∀ z ∈ S, ∃ x y : ℕ, is_valid_number x ∧ is_valid_number y ∧ 
    y = reverse_digits x ∧ z = Int.natAbs (x - y)) ∧ S.card = 90 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_z_values_l1252_125205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_school_activities_l1252_125282

theorem school_activities (students : Finset ℕ) 
  (chorus band drama : Finset ℕ) 
  (h_total : students.card = 1500)
  (h_chorus : chorus.card = 420)
  (h_band : band.card = 780)
  (h_chorus_band : (chorus ∩ band).card = 150)
  (h_drama : drama.card = 300)
  (h_drama_others : (drama ∩ (chorus ∪ band)).card = 50)
  (h_subset_chorus : chorus ⊆ students)
  (h_subset_band : band ⊆ students)
  (h_subset_drama : drama ⊆ students) :
  (students \ (chorus ∪ band ∪ drama)).card = 200 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_school_activities_l1252_125282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_is_four_digits_l1252_125278

/-- A digit is a natural number between 0 and 9, inclusive. -/
def Digit : Type := {n : ℕ // n ≤ 9}

/-- An even digit is a digit that is divisible by 2. -/
def EvenDigit : Type := {d : Digit // d.val % 2 = 0}

/-- Convert a three-digit number represented by its digits to a natural number. -/
def threeDigitToNat (a b c : Digit) : ℕ := 100 * a.val + 10 * b.val + c.val

/-- Convert a two-digit number represented by its digits to a natural number. -/
def twoDigitToNat (a b : Digit) : ℕ := 10 * a.val + b.val

/-- Convert a natural number to a Digit if it's less than or equal to 9. -/
def natToDigit (n : ℕ) : Option Digit :=
  if h : n ≤ 9 then some ⟨n, h⟩ else none

/-- The main theorem stating that the sum of 8765, C43, and D2 is always a 4-digit number,
    where C is a non-zero digit and D is an even digit. -/
theorem sum_is_four_digits (C : Digit) (hC : C.val ≠ 0) (D : EvenDigit) :
  ∃ (n : ℕ), n ≥ 1000 ∧ n < 10000 ∧
  n = 8765 + threeDigitToNat C ⟨4, by norm_num⟩ ⟨3, by norm_num⟩ + twoDigitToNat D.val ⟨2, by norm_num⟩ := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_is_four_digits_l1252_125278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_marbles_l1252_125201

/-- Represents the number of marbles Alvin has at each stage -/
def marbles : ℕ → ℕ := sorry

/-- Represents the number of stickers Alvin has at each stage -/
def stickers : ℕ → ℕ := sorry

/-- The exchange rate of marbles to stickers -/
def exchange_rate : ℕ := 5

/-- The initial number of marbles Alvin has -/
axiom initial_marbles : marbles 0 = 57

/-- The result of the first game -/
axiom first_game : marbles 1 = marbles 0 - 18

/-- The result of the second game -/
axiom second_game : marbles 2 = marbles 1 + 25

/-- The exchange of marbles for stickers after the second game -/
axiom exchange_after_second : marbles 3 = marbles 2 - 10 ∧ stickers 3 = stickers 2 + 2

/-- The result of the third game -/
axiom third_game : marbles 4 = marbles 3 - 12

/-- The result of the fourth game -/
axiom fourth_game : marbles 5 = marbles 4 + 15

/-- Giving away marbles to a friend -/
axiom give_away : marbles 6 = marbles 5 - 10

/-- Receiving marbles from another friend -/
axiom receive : marbles 7 = marbles 6 + 8

/-- Trading stickers for marbles -/
axiom final_trade : marbles 8 = marbles 7 + 20 ∧ stickers 8 = stickers 7 - 4

/-- The theorem stating that Alvin ends up with 75 marbles -/
theorem final_marbles : marbles 8 = 75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_marbles_l1252_125201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_sum_plus_ten_l1252_125296

theorem remainder_sum_plus_ten (a b c d : ℕ) 
  (ha : a % 53 = 33)
  (hb : b % 53 = 15)
  (hc : c % 53 = 27)
  (hd : d % 53 = 8) :
  (a + b + c + d + 10) % 53 = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_sum_plus_ten_l1252_125296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_types_have_equidistant_point_l1252_125207

-- Define the types of quadrilaterals
inductive QuadrilateralType
  | Square
  | Rectangle
  | Rhombus
  | Parallelogram
  | Kite

-- Define a property for having a point equidistant from all vertices
def has_equidistant_point (q : QuadrilateralType) : Bool :=
  match q with
  | QuadrilateralType.Square => true
  | QuadrilateralType.Rectangle => true
  | QuadrilateralType.Rhombus => false
  | QuadrilateralType.Parallelogram => false
  | QuadrilateralType.Kite => true

-- Define the list of quadrilateral types
def quadrilateral_types : List QuadrilateralType :=
  [QuadrilateralType.Square, QuadrilateralType.Rectangle, QuadrilateralType.Rhombus,
   QuadrilateralType.Parallelogram, QuadrilateralType.Kite]

-- Theorem stating that exactly 3 types have the equidistant point property
theorem three_types_have_equidistant_point :
  (quadrilateral_types.filter has_equidistant_point).length = 3 := by
  -- Evaluate the filter and check the length
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_types_have_equidistant_point_l1252_125207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_integers_with_sum_and_gcd_l1252_125210

theorem two_integers_with_sum_and_gcd (a b : ℕ) : 
  a + b = 104055 → Nat.gcd a b = 6937 → 
  (a, b) ∈ ({(6937, 79118), (13874, 90181), (27748, 76307), (48559, 55496)} : Set (ℕ × ℕ)) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_integers_with_sum_and_gcd_l1252_125210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_slope_l1252_125245

/-- Given an ellipse mx^2 + ny^2 = 1 (m > 0, n > 0) intersecting with the line y = 1 - x 
    at points A and B, if the slope of the line passing through the origin and the midpoint 
    of AB is √2/2, then n/m = √2 -/
theorem ellipse_intersection_slope (m n : ℝ) (A B : ℝ × ℝ) :
  m > 0 →
  n > 0 →
  m * A.1^2 + n * A.2^2 = 1 →
  m * B.1^2 + n * B.2^2 = 1 →
  A.2 = 1 - A.1 →
  B.2 = 1 - B.1 →
  (B.2 - A.2) / (B.1 - A.1) = -1 →
  ((A.2 + B.2) / 2) / ((A.1 + B.1) / 2) = Real.sqrt 2 / 2 →
  n / m = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_slope_l1252_125245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_l1252_125228

def line1 (t : ℝ) (b : ℝ) : ℝ × ℝ × ℝ := (3 + t * b, -2 + t * (-3), 0 + t * 2)
def line2 (u : ℝ) : ℝ × ℝ × ℝ := (2 + u * 2, -1 + u * 3, -6 + u * 4)

def direction_vector1 (b : ℝ) : ℝ × ℝ × ℝ := (b, -3, 2)
def direction_vector2 : ℝ × ℝ × ℝ := (2, 3, 4)

def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  let (x1, y1, z1) := v1
  let (x2, y2, z2) := v2
  x1 * x2 + y1 * y2 + z1 * z2

theorem perpendicular_lines (b : ℝ) :
  dot_product (direction_vector1 b) direction_vector2 = 0 ↔ b = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_l1252_125228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_wins_l1252_125219

/-- A game on a complete graph with 2000 vertices -/
structure Game where
  n : ℕ
  h : n = 2000

/-- The number of edges removed by the first player per turn -/
def first_player_edges : ℕ := 1

/-- The minimum number of edges removed by the second player per turn -/
def second_player_min_edges : ℕ := 2

/-- The maximum number of edges removed by the second player per turn -/
def second_player_max_edges : ℕ := 3

/-- Predicate to check if a vertex is isolated -/
def is_isolated (g : Game) (v : Fin g.n) : Prop := sorry

/-- A strategy is winning if it guarantees the player doesn't isolate a vertex -/
def is_winning_strategy (strategy : Game → ℕ → Bool) : Prop :=
  ∀ g : Game, ∃ move : ℕ, strategy g move ∧ 
    (move ≥ second_player_min_edges ∧ move ≤ second_player_max_edges) ∧
    ¬ (∃ v : Fin g.n, is_isolated g v)

/-- The main theorem: the second player has a winning strategy -/
theorem second_player_wins : 
  ∃ strategy : Game → ℕ → Bool, is_winning_strategy strategy := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_wins_l1252_125219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_calculation_volume_correct_l1252_125256

/-- The volume of the body bounded by the surfaces z = 4x^2 + 9y^2 and z = 6 -/
noncomputable def volume_paraboloid : ℝ := 3 * Real.pi

/-- The equation of the paraboloid surface -/
def paraboloid (x y z : ℝ) : Prop := z = 4 * x^2 + 9 * y^2

/-- The upper bounding plane -/
def upper_bound (z : ℝ) : Prop := z = 6

/-- The theorem stating the volume calculation as an integral -/
theorem volume_calculation :
  volume_paraboloid = ∫ z in (0 : ℝ)..(6 : ℝ), π * z / 6 := by
  sorry

/-- The theorem confirming the correct value of the volume -/
theorem volume_correct :
  volume_paraboloid = 3 * Real.pi := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_calculation_volume_correct_l1252_125256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shadow_preserves_ratio_l1252_125215

/-- Two line segments on a line -/
structure Segment where
  start : ℝ
  stop : ℝ

/-- A line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Ratio between two real numbers -/
noncomputable def ratio (a b : ℝ) : ℝ := a / b

/-- Shadow of a segment on another line -/
noncomputable def shadow (s : Segment) (l₁ l₂ : Line) : Segment :=
  sorry

/-- The main theorem -/
theorem shadow_preserves_ratio (s₁ s₂ : Segment) (l₁ l₂ : Line) :
  ratio s₁.stop s₁.start = 5 / 7 →
  ratio s₂.stop s₂.start = 7 / 5 →
  let shadow₁ := shadow s₁ l₁ l₂
  let shadow₂ := shadow s₂ l₁ l₂
  ratio shadow₁.stop shadow₁.start = 5 / 7 ∧
  ratio shadow₂.stop shadow₂.start = 7 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shadow_preserves_ratio_l1252_125215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tension_limit_tension_zero_tension_increases_l1252_125280

/-- The tension in a rope over a frictionless pulley with masses M and M + m -/
noncomputable def tension (M m : ℝ) : ℝ := 2 * M * (M + m) / (2 * M + m)

theorem tension_limit (M : ℝ) (h : M > 0) :
  ∀ ε > 0, ∃ N : ℝ, ∀ m ≥ N, |tension M m - 2 * M| < ε := by
  sorry

theorem tension_zero (M : ℝ) (h : M > 0) :
  tension M 0 = M := by
  sorry

theorem tension_increases (M : ℝ) (h : M > 0) :
  ∀ m₁ m₂ : ℝ, m₁ < m₂ → tension M m₁ < tension M m₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tension_limit_tension_zero_tension_increases_l1252_125280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_composition_inverse_value_l1252_125252

/-- Given linear functions f, g, and h, where h is the composition of f and g,
    and h^(-1) is known, prove that 2a - b equals -54/5. -/
theorem composition_inverse_value (a b : ℝ) (f g h : ℝ → ℝ) :
  (∀ x, f x = a * x + b) →
  (∀ x, g x = -5 * x + 7) →
  (∀ x, h x = f (g x)) →
  (∀ x, Function.invFun h x = x - 9) →
  2 * a - b = -54 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_composition_inverse_value_l1252_125252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_equidistant_implies_b_value_l1252_125290

noncomputable section

-- Define the function f(x)
def f (a b : ℝ) (x : ℝ) : ℝ := (1/12) * x^2 + a * x + b

-- Define the points A, B, C, and T
structure Point where
  x : ℝ
  y : ℝ

def T : Point := ⟨3, 3⟩

-- State the theorem
theorem intersection_and_equidistant_implies_b_value
  (a b : ℝ)
  (A C : Point)
  (B : Point)
  (h1 : f a b A.x = 0)
  (h2 : f a b C.x = 0)
  (h3 : B.x = 0 ∧ B.y = b)
  (h4 : (T.x - A.x)^2 + (T.y - A.y)^2 = (T.x - B.x)^2 + (T.y - B.y)^2)
  (h5 : (T.x - B.x)^2 + (T.y - B.y)^2 = (T.x - C.x)^2 + (T.y - C.y)^2)
  : b = -6 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_equidistant_implies_b_value_l1252_125290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_distance_l1252_125250

/-- Given 8 equally spaced trees along a straight road, where the distance between
    the first and fifth tree is 100 feet, the distance between the first and last
    tree is 175 feet. -/
theorem tree_distance (n : ℕ) (d : ℝ) (h1 : n = 8) (h2 : d = 100) :
  (d * (n - 1 : ℝ) / 4) = 175 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_distance_l1252_125250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_group_frequency_l1252_125297

theorem sixth_group_frequency 
  (sample_size : ℕ) 
  (num_groups : ℕ) 
  (group1 : ℕ) 
  (group2 : ℕ) 
  (group3 : ℕ) 
  (group4 : ℕ) 
  (group5_ratio : ℚ) : 
  sample_size = 40 → 
  num_groups = 6 → 
  group1 = 5 → 
  group2 = 6 → 
  group3 = 7 → 
  group4 = 10 → 
  group5_ratio = 1/5 → 
  ∃ (group6 : ℕ), group6 = sample_size - group1 - group2 - group3 - group4 - (group5_ratio * ↑sample_size).floor ∧ group6 = 4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_group_frequency_l1252_125297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_identification_l1252_125249

-- Define the function f as noncomputable
noncomputable def f (ω : ℝ) (φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

-- State the theorem
theorem function_identification 
  (ω : ℝ) 
  (φ : ℝ) 
  (h_ω_pos : ω > 0)
  (h_φ_bound : |φ| < π / 2)
  (h_max : f ω φ (π / 4) = 1)
  (h_min : f ω φ (7 * π / 12) = -1) :
  ∀ x, f ω φ x = Real.sin (3 * x - π / 4) :=
by
  sorry -- Skip the proof for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_identification_l1252_125249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l1252_125244

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the domain of f(x+1)
def domain_f_shifted : Set ℝ := Set.Ioo (-2) 2

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := f x / Real.sqrt x

-- Theorem statement
theorem domain_of_g :
  (∀ x, x + 1 ∈ domain_f_shifted → f (x + 1) = f (x + 1)) →
  {x : ℝ | g x = g x} = Set.Ioo 0 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l1252_125244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhomboid_toothpicks_count_l1252_125206

/-- Represents a rhomboid shape constructed from small equilateral triangles --/
structure Rhomboid where
  base_triangles : ℕ
  toothpicks_per_triangle : ℚ

/-- Calculates the number of toothpicks required to construct the rhomboid --/
def toothpicks_required (r : Rhomboid) : ℕ :=
  (r.base_triangles * (r.base_triangles + 1) * r.toothpicks_per_triangle).floor.toNat

/-- Theorem stating the number of toothpicks required for a specific rhomboid --/
theorem rhomboid_toothpicks_count :
  ∃ (r : Rhomboid), r.base_triangles = 987 ∧ r.toothpicks_per_triangle = 3/2 ∧ toothpicks_required r = 1463598 := by
  sorry

#eval toothpicks_required { base_triangles := 987, toothpicks_per_triangle := 3/2 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhomboid_toothpicks_count_l1252_125206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_sum_mod_37_l1252_125285

theorem remainder_sum_mod_37 (a b c : ℕ) 
  (ha : a % 37 = 15)
  (hb : b % 37 = 22)
  (hc : c % 37 = 7) :
  (a + b + c) % 37 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_sum_mod_37_l1252_125285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wedge_volume_equals_200pi_over_3_l1252_125251

/-- The volume of a wedge that covers one-third of a cylinder with radius 5 cm and height 8 cm -/
noncomputable def wedge_volume : ℝ :=
  let cylinder_radius : ℝ := 5
  let cylinder_height : ℝ := 8
  let cylinder_volume : ℝ := Real.pi * cylinder_radius ^ 2 * cylinder_height
  (1 / 3) * cylinder_volume

theorem wedge_volume_equals_200pi_over_3 :
  wedge_volume = (200 * Real.pi) / 3 := by
  unfold wedge_volume
  simp [Real.pi]
  ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wedge_volume_equals_200pi_over_3_l1252_125251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_proof_l1252_125238

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

-- Define the inverse function g
noncomputable def g (x : ℝ) : ℝ := Real.log x / Real.log 3

theorem inverse_function_proof (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a 2 = 9) :
  Function.LeftInverse g (f 3) ∧ Function.RightInverse g (f 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_proof_l1252_125238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_income_l1252_125265

/-- Calculates the annual income from a stock investment --/
noncomputable def annual_income (investment : ℝ) (dividend_rate : ℝ) (market_price : ℝ) (nominal_value : ℝ) : ℝ :=
  (investment / market_price) * (dividend_rate * nominal_value)

/-- Theorem: The annual income from investing $6800 in a 40% stock at $136 per share is $2000 --/
theorem investment_income :
  let investment : ℝ := 6800
  let dividend_rate : ℝ := 0.4
  let market_price : ℝ := 136
  let nominal_value : ℝ := 100
  annual_income investment dividend_rate market_price nominal_value = 2000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_income_l1252_125265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l1252_125231

-- Define the functions
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a
noncomputable def h (a : ℝ) (x : ℝ) : ℝ := x^a

-- State the theorem
theorem function_inequality (a : ℝ) (ha : 0 < a) (ha' : a < 1) :
  h a 2 > f a 2 ∧ f a 2 > g a 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l1252_125231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1252_125260

noncomputable def f (a b x : ℝ) : ℝ := a * Real.sin (2 * x - Real.pi / 3) + b

noncomputable def g (a b x : ℝ) : ℝ := b * Real.cos (a * x + Real.pi / 6)

theorem function_properties (a b : ℝ) (ha : a > 0) 
  (hmax : ∀ x, f a b x ≤ 1) (hmin : ∀ x, f a b x ≥ -5)
  (hmax_exists : ∃ x, f a b x = 1) (hmin_exists : ∃ x, f a b x = -5) :
  a = 3 ∧ b = -2 ∧ 
  (∀ x, g a b x ≤ 2) ∧
  (∀ k : ℤ, g a b (5 * Real.pi / 18 + 2 * k * Real.pi / 3) = 2) ∧
  (∀ x, g a b x = 2 → ∃ k : ℤ, x = 5 * Real.pi / 18 + 2 * k * Real.pi / 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1252_125260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_annual_cereal_spending_correct_l1252_125287

/-- Calculates Travis's annual cereal spending --/
def annual_cereal_spending (box_a_cost box_b_cost box_c_cost box_d_cost box_e_cost : ℝ) : ℝ :=
  let week1_cost := box_a_cost + (box_b_cost * 0.5) + (box_c_cost * 0.25) + (box_d_cost * 0.75) + (box_e_cost * 1.5)
  let week2_cost := week1_cost * 0.8
  let week3_cost := box_a_cost + (box_c_cost * 0.25) + (box_d_cost * 0.75) + (box_e_cost * 1.5)
  let week4_cost := box_a_cost + (box_b_cost * 0.5) + (box_c_cost * 0.25) + (box_d_cost * 0.75) + (box_e_cost * 1.5 * 0.85)
  let monthly_cost := week1_cost + week2_cost + week3_cost + week4_cost
  monthly_cost * 12

theorem annual_cereal_spending_correct :
  annual_cereal_spending 2.5 3.5 4.0 5.25 6.0 = 792.24 := by
  sorry

#eval annual_cereal_spending 2.5 3.5 4.0 5.25 6.0

end NUMINAMATH_CALUDE_ERRORFEEDBACK_annual_cereal_spending_correct_l1252_125287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_equation_ratio_l1252_125242

theorem sqrt_equation_ratio (a b c : ℕ) (h1 : c < 90) (h2 : 0 < c) 
  (h3 : Real.sqrt (9 - 8 * Real.sin (50 * π / 180)) = (a : ℝ) + (b : ℝ) * Real.sin (c * π / 180)) :
  (b + c) / a = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_equation_ratio_l1252_125242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_polynomials_l1252_125236

/-- An algebraic expression -/
inductive AlgebraicExpr
| reciprocal_x : AlgebraicExpr
| linear_xy : AlgebraicExpr
| frac_a2b : AlgebraicExpr
| frac_xy_pi : AlgebraicExpr
| frac_5y_4x : AlgebraicExpr
| zero : AlgebraicExpr

/-- Predicate to determine if an algebraic expression is a polynomial -/
def is_polynomial : AlgebraicExpr → Bool
| AlgebraicExpr.reciprocal_x => false
| AlgebraicExpr.linear_xy => true
| AlgebraicExpr.frac_a2b => true
| AlgebraicExpr.frac_xy_pi => true
| AlgebraicExpr.frac_5y_4x => false
| AlgebraicExpr.zero => true

/-- The list of all given algebraic expressions -/
def all_expressions : List AlgebraicExpr :=
  [AlgebraicExpr.reciprocal_x, AlgebraicExpr.linear_xy, AlgebraicExpr.frac_a2b,
   AlgebraicExpr.frac_xy_pi, AlgebraicExpr.frac_5y_4x, AlgebraicExpr.zero]

theorem count_polynomials :
  (all_expressions.filter is_polynomial).length = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_polynomials_l1252_125236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overall_average_marks_l1252_125279

/-- Represents a class of students with their average marks -/
structure StudentClass where
  students : ℕ
  average : ℝ

/-- Represents a class with subgroups -/
structure ClassWithSubgroups where
  students : ℕ
  weightedAverage : ℝ
  subgroup1 : StudentClass
  subgroup2 : StudentClass

theorem overall_average_marks 
  (class1 : StudentClass)
  (class2 : StudentClass)
  (class3 : StudentClass)
  (class4 : ClassWithSubgroups)
  (h1 : class1.students = 30 ∧ class1.average = 50)
  (h2 : class2.students = 45 ∧ class2.average = 70)
  (h3 : class3.students = 25 ∧ class3.average = 80)
  (h4 : class4.students = 60 ∧ class4.weightedAverage = 55)
  (h5 : class4.subgroup1.students = 20 ∧ class4.subgroup1.average = 48)
  (h6 : class4.subgroup2.students = 40 ∧ class4.subgroup2.average = 58)
  (h7 : class4.students = class4.subgroup1.students + class4.subgroup2.students) :
  let totalStudents := class1.students + class2.students + class3.students + class4.students
  let totalMarks := class1.students * class1.average + 
                    class2.students * class2.average + 
                    class3.students * class3.average + 
                    class4.subgroup1.students * class4.subgroup1.average +
                    class4.subgroup2.students * class4.subgroup2.average
  totalMarks / totalStudents = 62.0625 := by
  sorry

#check overall_average_marks

end NUMINAMATH_CALUDE_ERRORFEEDBACK_overall_average_marks_l1252_125279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_B_speed_l1252_125202

/-- Represents the speeds and positions of two cars on a road --/
structure CarProblem where
  initial_distance : ℚ  -- Initial distance between cars (miles)
  speed_A : ℚ          -- Speed of Car A (mph)
  time : ℚ              -- Time taken for Car A to overtake Car B (hours)
  final_distance : ℚ   -- Final distance between cars after overtaking (miles)

/-- Calculates the speed of Car B given the problem parameters --/
def speed_of_car_B (p : CarProblem) : ℚ :=
  (p.speed_A * p.time - p.initial_distance - p.final_distance) / p.time

/-- Theorem stating that given the problem conditions, the speed of Car B is 50 mph --/
theorem car_B_speed (p : CarProblem) 
  (h1 : p.initial_distance = 40)
  (h2 : p.speed_A = 58)
  (h3 : p.time = 6)
  (h4 : p.final_distance = 8) : 
  speed_of_car_B p = 50 := by
  sorry

#eval speed_of_car_B { initial_distance := 40, speed_A := 58, time := 6, final_distance := 8 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_B_speed_l1252_125202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_side_length_l1252_125246

/-- A rhombus with area K and one diagonal twice the length of the other has side length √(5K)/2 -/
theorem rhombus_side_length (K : ℝ) (K_pos : K > 0) : 
  ∃ (d₁ d₂ s : ℝ), 
    d₁ > 0 ∧ d₂ > 0 ∧ s > 0 ∧
    d₂ = 2 * d₁ ∧ 
    K = (1/2) * d₁ * d₂ ∧
    s = Real.sqrt (5*K) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_side_length_l1252_125246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_sum_division_l1252_125209

theorem factorial_sum_division (n : ℕ) : (n + 1).factorial + (n + 2).factorial = 80 * n.factorial :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_sum_division_l1252_125209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spheres_from_cylinder_l1252_125216

/-- The number of spheres that can be made from a cylinder -/
noncomputable def num_spheres (cylinder_diameter : ℝ) (cylinder_height : ℝ) (sphere_diameter : ℝ) : ℝ :=
  (Real.pi * (cylinder_diameter / 2)^2 * cylinder_height) / ((4/3) * Real.pi * (sphere_diameter / 2)^3)

/-- Theorem: The number of spheres with diameter 8 cm that can be made from a cylinder 
    with diameter 16 cm and height 16 cm is equal to 12 -/
theorem spheres_from_cylinder :
  num_spheres 16 16 8 = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_spheres_from_cylinder_l1252_125216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increasing_interval_l1252_125253

/-- The function f(x) defined as 2sin(x)cos(x) - 2cos²(x) + 1 -/
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos x - 2 * (Real.cos x)^2 + 1

/-- Theorem stating that the monotonic increasing interval of f(x) is (kπ - π/8, kπ + 3π/8) where k ∈ ℤ -/
theorem f_monotonic_increasing_interval :
  ∀ k : ℤ, StrictMonoOn f (Set.Ioo ((k : ℝ) * π - π / 8) ((k : ℝ) * π + 3 * π / 8)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increasing_interval_l1252_125253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_product_l1252_125299

/-- The ellipse equation -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 12 = 1

/-- The foci of the ellipse -/
def left_focus : ℝ × ℝ := (-2, 0)
def right_focus : ℝ × ℝ := (2, 0)

/-- The theorem to prove -/
theorem ellipse_foci_product (P : ℝ × ℝ) 
  (h_P_on_ellipse : is_on_ellipse P.1 P.2)
  (h_dot_product : (P.1 - left_focus.1) * (P.1 - right_focus.1) + 
                   (P.2 - left_focus.2) * (P.2 - right_focus.2) = 9) :
  ((P.1 - left_focus.1)^2 + (P.2 - left_focus.2)^2) * 
  ((P.1 - right_focus.1)^2 + (P.2 - right_focus.2)^2) = 225 := by
  sorry

/-- A helper lemma for the main theorem -/
lemma distance_product_eq_225 (P : ℝ × ℝ) 
  (h_P_on_ellipse : is_on_ellipse P.1 P.2)
  (h_dot_product : (P.1 - left_focus.1) * (P.1 - right_focus.1) + 
                   (P.2 - left_focus.2) * (P.2 - right_focus.2) = 9) :
  ((P.1 - left_focus.1)^2 + (P.2 - left_focus.2)^2) * 
  ((P.1 - right_focus.1)^2 + (P.2 - right_focus.2)^2) = 225 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_product_l1252_125299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_intersecting_lines_l1252_125200

/-- A line that intersects the circle x^2 + y^2 = 100 at an integer point -/
structure IntersectingLine where
  a : ℤ
  b : ℤ
  x : ℤ
  y : ℤ
  line_eq : a * x + b * y = 2017
  circle_eq : x^2 + y^2 = 100

/-- The set of all intersecting lines -/
def intersecting_lines : Set IntersectingLine :=
  {l : IntersectingLine | True}

/-- Finite type instance for intersecting_lines -/
instance : Fintype intersecting_lines :=
  sorry

/-- The number of intersecting lines is 24 -/
theorem count_intersecting_lines : Fintype.card intersecting_lines = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_intersecting_lines_l1252_125200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_range_circle_through_focus_l1252_125243

-- Define the line and parabola
def line (a : ℝ) (x : ℝ) : ℝ := a * x + 1
def parabola (x : ℝ) : ℝ := 4 * x

-- Define the intersection points
def intersectionPoints (a : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ x, p.1 = x ∧ p.2 = line a x ∧ p.2^2 = parabola p.1}

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Theorem for the range of a
theorem intersection_range :
  {a : ℝ | ∃ x y, (x, y) ∈ intersectionPoints a} = { a | a < 0 ∨ (a > 0 ∧ a < 1) } := by sorry

-- Function to check if a circle passes through a point
def circlePassesThroughPoint (center : ℝ × ℝ) (radius : ℝ) (point : ℝ × ℝ) : Prop :=
  (point.1 - center.1)^2 + (point.2 - center.2)^2 = radius^2

-- Theorem for the values of a when the circle passes through the focus
theorem circle_through_focus (a : ℝ) :
  (∃ A B : ℝ × ℝ, A ∈ intersectionPoints a ∧ B ∈ intersectionPoints a ∧
    let center := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
    let radius := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) / 2
    circlePassesThroughPoint center radius focus) ↔
  (a = -3 - 2 * Real.sqrt 3 ∨ a = -3 + 2 * Real.sqrt 3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_range_circle_through_focus_l1252_125243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_difference_l1252_125289

/-- The circumference of the smaller circle in meters -/
def small_circumference : ℝ := 396

/-- The circumference of the larger circle in meters -/
def large_circumference : ℝ := 704

/-- The value of pi -/
noncomputable def π : ℝ := Real.pi

/-- Calculate the radius of a circle given its circumference -/
noncomputable def radius (circumference : ℝ) : ℝ :=
  circumference / (2 * π)

/-- Calculate the area of a circle given its radius -/
noncomputable def area (radius : ℝ) : ℝ :=
  π * radius^2

/-- The statement to be proved -/
theorem circle_area_difference :
  let small_radius := radius small_circumference
  let large_radius := radius large_circumference
  let small_area := area small_radius
  let large_area := area large_radius
  abs ((large_area - small_area) - 26948.4) < 0.1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_difference_l1252_125289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midsegment_divides_area_l1252_125220

/-- Represents a trapezoid with parallel bases of lengths a and b -/
structure Trapezoid where
  a : ℝ
  b : ℝ
  h : 0 < a ∧ 0 < b

/-- The length of the midsegment of a trapezoid that divides it into two equal areas -/
noncomputable def midsegment_length (t : Trapezoid) : ℝ :=
  Real.sqrt ((t.a ^ 2 + t.b ^ 2) / 2)

/-- Theorem: The midsegment length divides the trapezoid into two equal areas -/
theorem midsegment_divides_area (t : Trapezoid) :
  midsegment_length t = Real.sqrt ((t.a ^ 2 + t.b ^ 2) / 2) := by
  -- Proof goes here
  sorry

#check midsegment_divides_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midsegment_divides_area_l1252_125220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_after_translation_l1252_125234

/-- Applies a translation to a point in a 2D plane -/
def translate (p : ℝ × ℝ) (dx dy : ℝ) : ℝ × ℝ :=
  (p.1 + dx, p.2 + dy)

theorem midpoint_after_translation :
  let B : ℝ × ℝ := (2, 2)
  let G : ℝ × ℝ := (6, 2)
  let dx : ℝ := -6  -- Translation left is negative
  let dy : ℝ := 3
  let B' := translate B dx dy
  let G' := translate G dx dy
  (B'.1 + G'.1) / 2 = -2 ∧ (B'.2 + G'.2) / 2 = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_after_translation_l1252_125234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1252_125274

noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 2 then (1/2) * x^2 + x
  else if -2 ≤ x ∧ x < 0 then (1/2) * x^2 - x
  else 0  -- This case should never occur given the domain

theorem f_properties :
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f x = f (-x)) ∧  -- f is even on [-2,2]
  (∀ x ∈ Set.Icc 0 2, f x = (1/2) * x^2 + x) →  -- f(x) = (1/2)x^2 + x for 0 ≤ x ≤ 2
  (∀ x ∈ Set.Icc (-2 : ℝ) 0, f x = (1/2) * x^2 - x) ∧  -- Analytical expression for -2 ≤ x ≤ 0
  (Set.Ioo 0 1) ⊆ {a : ℝ | f (a + 1) - f (2 * a - 1) > 0} ∧  -- Range of a
  {a : ℝ | f (a + 1) - f (2 * a - 1) > 0} ⊆ (Set.Ioc 0 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1252_125274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prop_1_prop_3_prop_4_prop_2_incorrect_l1252_125294

-- Define the basic geometric objects
structure Point : Type
structure Line : Type
structure Plane : Type

-- Define the relations between geometric objects
axiom contains : Plane → Line → Prop
axiom contains_point : Plane → Point → Prop
axiom parallel : Plane → Plane → Prop
axiom parallel_lines : Line → Line → Prop
axiom skew : Line → Line → Prop

-- Define distance functions
axiom distance_line_plane : Line → Plane → ℝ
axiom distance_plane_plane : Plane → Plane → ℝ
axiom distance_line_line : Line → Line → ℝ
axiom distance_point_plane : Point → Plane → ℝ

-- Theorem for proposition ①
theorem prop_1 (a : Line) (α β : Plane) :
  contains β a → parallel α β → distance_line_plane a α = distance_plane_plane α β := by sorry

-- Theorem for proposition ③
theorem prop_3 (a b : Line) (α β : Plane) :
  skew a b → contains α a → contains β b → parallel α β →
  distance_line_line a b = distance_plane_plane α β := by sorry

-- Theorem for proposition ④
theorem prop_4 (A : Point) (α β : Plane) :
  contains_point α A → parallel α β → distance_point_plane A β = distance_plane_plane α β := by sorry

-- Theorem for the incorrectness of proposition ②
theorem prop_2_incorrect :
  ∃ (l1 l2 : Line) (α β : Plane),
    parallel_lines l1 l2 → contains α l1 → contains β l2 → parallel α β →
    distance_line_line l1 l2 ≠ distance_plane_plane α β := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prop_1_prop_3_prop_4_prop_2_incorrect_l1252_125294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_heron_l1252_125254

/-- Given a triangle with sides in ratio 3:4:5 inscribed in a circle of radius 5,
    its area calculated using Heron's formula is 24 square units. -/
theorem triangle_area_heron (a b c : ℝ) (r : ℝ) :
  r = 5 →
  (∃ (k : ℝ), k > 0 ∧ a = 3 * k ∧ b = 4 * k ∧ c = 5 * k) →
  c = 2 * r →
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c)) = 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_heron_l1252_125254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_other_side_length_l1252_125288

/-- The area of a trapezium given the lengths of its parallel sides and the distance between them. -/
noncomputable def trapeziumArea (a b h : ℝ) : ℝ := (a + b) * h / 2

/-- Theorem: Given a trapezium with one parallel side of 18 cm, a distance between parallel sides
    of 16 cm, and an area of 304 square centimeters, the length of the other parallel side is 20 cm. -/
theorem trapezium_other_side_length :
  ∃ (x : ℝ), trapeziumArea 18 x 16 = 304 ∧ x = 20 := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_other_side_length_l1252_125288
