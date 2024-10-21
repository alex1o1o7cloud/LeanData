import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_cone_volume_formula_l588_58845

open Real

variable (l α β : ℝ)

/-- The volume of the common part of two cones with a common height, where one cone has a slant
    height l forming an angle α with the height, and the other cone's slant height forms an angle β
    with the height. -/
noncomputable def commonConeVolume (l α β : ℝ) : ℝ :=
  (Real.pi * l^3 * (sin (2 * α))^2 * cos α * sin β^2) /
  (12 * (sin (α + β))^2)

/-- Theorem stating that the volume of the common part of two cones with the given properties
    is equal to the formula derived. -/
theorem common_cone_volume_formula (l α β : ℝ) (h1 : 0 < l) (h2 : 0 < α) (h3 : 0 < β) (h4 : α + β < π) :
  commonConeVolume l α β = (Real.pi * l^3 * (sin (2 * α))^2 * cos α * sin β^2) /
                           (12 * (sin (α + β))^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_cone_volume_formula_l588_58845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l588_58837

noncomputable section

-- Define the ellipse C
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the eccentricity
def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - b^2 / a^2)

-- Define the distance between two points
def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

theorem ellipse_properties :
  ∀ a b : ℝ,
  a > b ∧ b > 0 →
  eccentricity a b = Real.sqrt 3 / 2 →
  ellipse a b (Real.sqrt 2) (Real.sqrt 2 / 2) →
  (∀ x y : ℝ, ellipse a b x y ↔ x^2 / 4 + y^2 = 1) ∧
  (∃ x₀ : ℝ,
    x₀ = 4 ∧
    ∀ k x₁ y₁ x₂ y₂ : ℝ,
    ellipse a b x₁ y₁ ∧ ellipse a b x₂ y₂ ∧
    y₁ = k * (x₁ - 1) ∧ y₂ = k * (x₂ - 1) →
    (x₀ - x₁) / (x₀ - x₂) = distance x₁ y₁ 1 0 / distance x₂ y₂ 1 0) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l588_58837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l588_58874

noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

def solution_set : Set ℝ := {1/10, 10^(Real.sqrt 3), 100}

theorem equation_solutions :
  ∀ x : ℝ, x > 0 → ((log10 x)^2 - (floor (log10 x) : ℝ) - 2 = 0 ↔ x ∈ solution_set) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l588_58874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_family_reunion_cost_l588_58822

/-- Calculates the total cost of meat and side dishes for Peter's family reunion --/
noncomputable def total_cost (chicken_weight : ℝ) (chicken_price hamburger_price hotdog_price sausage_price side_dish_price : ℝ) : ℝ :=
  let hamburger_weight := chicken_weight / 2
  let hotdog_weight := hamburger_weight + 2
  let sausage_weight := 1.5 * hamburger_weight
  let side_dish_weight := (chicken_weight + hamburger_weight + sausage_weight) / 2
  chicken_weight * chicken_price +
  hamburger_weight * hamburger_price +
  hotdog_weight * hotdog_price +
  sausage_weight * sausage_price +
  side_dish_weight * side_dish_price

/-- Theorem stating that the total cost for Peter's family reunion is $212 --/
theorem family_reunion_cost : 
  total_cost 16 4 5 3 3.5 2 = 212 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_family_reunion_cost_l588_58822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2007th_term_l588_58860

theorem sequence_2007th_term :
  ∃ a : ℕ → ℕ,
    a 1 = 2 ∧
    (∀ n : ℕ, n ≥ 1 → a n = 2^n) ∧
    a 2007 = 2^2007 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2007th_term_l588_58860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_exponential_function_l588_58868

theorem range_of_exponential_function (x : ℝ) :
  (2 : ℝ)^(x^2 + 1) ≤ ((1/4) : ℝ)^(x - 2) →
  ∃ (y : ℝ), y = (2 : ℝ)^x ∧ y ∈ Set.Icc (1/8 : ℝ) 2 ∧
  ∀ (z : ℝ), z = (2 : ℝ)^x → z ∈ Set.Icc (1/8 : ℝ) 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_exponential_function_l588_58868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_people_needed_l588_58850

/-- The number of person-hours required to paint the fence -/
def personHours : ℚ := 8 * 3

/-- The time (in hours) it takes to paint the fence with the new number of people -/
def newTime : ℚ := 2

/-- The initial number of people painting the fence -/
def initialPeople : ℕ := 8

/-- Calculates the total number of people needed to paint the fence in the new time -/
def totalPeopleNeeded : ℕ := Int.natAbs (Int.floor (personHours / newTime))

/-- The number of additional people needed -/
def additionalPeople : ℕ := totalPeopleNeeded - initialPeople

theorem additional_people_needed :
  additionalPeople = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_people_needed_l588_58850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_two_is_zero_l588_58806

theorem sum_of_two_is_zero 
  (x y z t : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (ht : t ≠ 0)
  (hsum : x + y + z + t = 0)
  (hrecip : 1/x + 1/y + 1/z + 1/t = 0) :
  ∃ (i j : ℝ), i ∈ ({x, y, z, t} : Set ℝ) ∧ j ∈ ({x, y, z, t} : Set ℝ) ∧ i ≠ j ∧ i + j = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_two_is_zero_l588_58806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_from_lateral_surface_l588_58824

-- Define the constants
noncomputable def sector_radius : ℝ := 2
noncomputable def central_angle : ℝ := 270 * (Real.pi / 180)  -- Convert to radians

-- Define the theorem
theorem cone_volume_from_lateral_surface (h : central_angle = 270 * (Real.pi / 180)) :
  let base_radius := sector_radius * (central_angle / (2 * Real.pi))
  let cone_height := Real.sqrt (sector_radius^2 - base_radius^2)
  (1 / 3) * Real.pi * base_radius^2 * cone_height = (3 * Real.sqrt 7) / 8 * Real.pi :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_from_lateral_surface_l588_58824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sailboat_max_power_speed_l588_58880

/-- The force equation for the air stream acting on the sail. -/
noncomputable def force (C S ρ v₀ v : ℝ) : ℝ := (C * S * ρ * (v₀ - v)^2) / 2

/-- The power exerted by the wind. -/
noncomputable def power (C S ρ v₀ v : ℝ) : ℝ := force C S ρ v₀ v * v

/-- Theorem stating the speed of the sailboat when the power is maximized. -/
theorem sailboat_max_power_speed (C S ρ v₀ : ℝ) (hC : C > 0) (hS : S > 0) (hρ : ρ > 0) (hv₀ : v₀ > 0) :
  ∃ v : ℝ, v > 0 ∧ v = v₀ / 3 ∧ 
  ∀ u : ℝ, u ≠ v → power C S ρ v₀ v ≥ power C S ρ v₀ u :=
by
  sorry

#check sailboat_max_power_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sailboat_max_power_speed_l588_58880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_separation_l588_58829

/-- The time it takes for Adam and Simon to be 50 miles apart -/
noncomputable def separation_time : ℝ := 50 / Real.sqrt 164

/-- Adam's speed in miles per hour -/
def adam_speed : ℝ := 10

/-- Simon's speed in miles per hour -/
def simon_speed : ℝ := 8

/-- The distance they are apart after separation_time hours -/
def separation_distance : ℝ := 50

theorem bicycle_separation :
  separation_distance = Real.sqrt ((adam_speed * separation_time)^2 + (simon_speed * separation_time)^2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_separation_l588_58829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_lambda_inequality_l588_58807

theorem largest_lambda_inequality (μ : ℝ) (hμ : μ ≥ 0) :
  (∀ a b c d : ℝ, a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 →
    a^2 + b^2 + c^2 + d^2 + μ*a*d ≥ a*b + (3/2)*b*c + c*d + μ*a*d) ∧
  (∀ lambda : ℝ, lambda > 3/2 →
    ∃ a b c d : ℝ, a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧
      a^2 + b^2 + c^2 + d^2 + μ*a*d < a*b + lambda*b*c + c*d + μ*a*d) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_lambda_inequality_l588_58807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_differ_by_3_l588_58815

/-- A standard 6-sided die -/
def StandardDie : Finset ℕ := Finset.range 6

/-- The set of all possible outcomes when rolling the die twice -/
def AllOutcomes : Finset (ℕ × ℕ) :=
  Finset.product StandardDie StandardDie

/-- The set of favorable outcomes (pairs differing by exactly 3) -/
def FavorableOutcomes : Finset (ℕ × ℕ) :=
  Finset.filter (fun p => (p.1 + 3 = p.2) ∨ (p.2 + 3 = p.1)) AllOutcomes

/-- The probability of rolling two numbers that differ by exactly 3 -/
def ProbabilityDifferBy3 : ℚ :=
  (FavorableOutcomes.card : ℚ) / (AllOutcomes.card : ℚ)

theorem probability_differ_by_3 :
  ProbabilityDifferBy3 = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_differ_by_3_l588_58815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l588_58804

/-- Definition of the ellipse C -/
noncomputable def ellipse (x y : ℝ) (a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- Definition of eccentricity -/
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - b^2 / a^2)

/-- Definition of the sum of distances to foci -/
def sum_distances_to_foci (a : ℝ) : ℝ := 2 * a

/-- Definition of symmetry with respect to y = 2x -/
def symmetric_points (x₀ y₀ x₁ y₁ : ℝ) : Prop :=
  (y₀ - y₁) / (x₀ - x₁) * 2 = -1 ∧ (y₀ + y₁) / 2 = 2 * ((x₀ + x₁) / 2)

theorem ellipse_properties (a b : ℝ) (h₁ : a > b) (h₂ : b > 0) :
  eccentricity a b = Real.sqrt 2 / 2 →
  sum_distances_to_foci a = 4 →
  (∃ x y, ellipse x y a b) →
  (∀ x₀ y₀ x₁ y₁, ellipse x₀ y₀ a b → symmetric_points x₀ y₀ x₁ y₁ →
    ∃ t, t = 3 * x₁ - 4 * y₁ ∧ -10 ≤ t ∧ t ≤ 10) →
  a = 2 ∧ b = Real.sqrt 2 ∧
  (∀ x y, ellipse x y a b ↔ x^2 / 4 + y^2 / 2 = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l588_58804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_half_angle_l588_58851

theorem sin_half_angle (θ : Real) 
  (h1 : Real.sin θ = 3/5) 
  (h2 : 5*Real.pi/2 < θ) 
  (h3 : θ < 3*Real.pi) : 
  Real.sin (θ/2) = -3*Real.sqrt 10/10 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_half_angle_l588_58851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_s_l588_58831

-- Define the function s(x) as noncomputable
noncomputable def s (x : ℝ) : ℝ := 1 / (2 - x)^3

-- State the theorem about the range of s(x)
theorem range_of_s :
  ∀ y : ℝ, y ≠ 0 → ∃ x : ℝ, x ≠ 2 ∧ s x = y :=
by
  -- The proof is skipped using 'sorry'
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_s_l588_58831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hydrochloric_acid_moles_l588_58893

/-- Represents the number of moles of a substance -/
def Moles : Type := ℝ

instance : OfNat Moles n where
  ofNat := (n : ℝ)

/-- Represents the chemical reaction between Sodium chloride and Nitric acid -/
structure Reaction where
  sodium_chloride : Moles
  nitric_acid : Moles
  sodium_nitrate : Moles
  hydrochloric_acid : Moles

/-- The theorem stating the number of moles of Hydrochloric acid formed in the reaction -/
theorem hydrochloric_acid_moles (r : Reaction) 
  (h1 : r.sodium_chloride = 2)
  (h2 : r.nitric_acid = 2)
  (h3 : r.sodium_nitrate = 2)
  (h4 : r.sodium_nitrate = r.hydrochloric_acid) :
  r.hydrochloric_acid = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hydrochloric_acid_moles_l588_58893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wang_purchases_l588_58866

-- Define the discount function
noncomputable def discount (price : ℝ) : ℝ :=
  if price < 200 then price
  else if price ≤ 500 then price * 0.9
  else 500 * 0.8 + (price - 500) * 0.5

-- Define Mr. Wang's purchases
def purchase1 : ℝ := 100
def purchase2 : ℝ := 432

-- Theorem statement
theorem wang_purchases :
  (discount purchase1 = purchase1) ∧
  ((discount purchase2 = 432 ∧ purchase2 = 480) ∨ 
   (500 * 0.8 + (purchase2 - 500) * 0.5 = 432 ∧ purchase2 = 564)) ∧
  ((discount (purchase1 + 480) = 440 ∧ 532 - 440 = 92) ∨
   (discount (purchase1 + 564) = 482 ∧ 532 - 482 = 50)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wang_purchases_l588_58866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2003_value_l588_58881

/-- Sequence definition -/
def a : ℕ → ℕ
  | 0 => 0  -- Add this case for 0
  | 1 => 0
  | n + 2 => a (n + 1) + 2 * (n + 1)

/-- Theorem statement -/
theorem a_2003_value : a 2003 = 2003 * 2002 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2003_value_l588_58881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_closer_to_origin_l588_58875

/-- The rectangular region from which point P is selected -/
def Rectangle : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 3 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2}

/-- The area of the rectangular region -/
def rectangleArea : ℝ := 6

/-- The region where points are closer to (0,0) than to (4,2) -/
def CloserRegion : Set (ℝ × ℝ) :=
  {p | p.1^2 + p.2^2 < (p.1 - 4)^2 + (p.2 - 2)^2}

/-- The intersection of the Rectangle and CloserRegion -/
def IntersectionRegion : Set (ℝ × ℝ) :=
  Rectangle ∩ CloserRegion

/-- The area of the IntersectionRegion -/
noncomputable def intersectionArea : ℝ := (3/2)

theorem probability_closer_to_origin :
  (intersectionArea / rectangleArea : ℝ) = 1/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_closer_to_origin_l588_58875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_eight_l588_58883

/-- The area of a triangle formed by two vectors -/
noncomputable def triangleArea (a b : ℝ × ℝ) : ℝ :=
  (1/2) * abs (a.1 * b.2 - a.2 * b.1)

/-- Theorem: The area of the triangle with vertices (0,0), (3,-2), and (-1,6) is 8 -/
theorem triangle_area_is_eight :
  triangleArea (3, -2) (-1, 6) = 8 := by
  -- Unfold the definition of triangleArea
  unfold triangleArea
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_eight_l588_58883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_equals_four_thirds_l588_58826

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2
  else if x ≤ 2 then 1
  else 0  -- This else case is added to make the function total

-- State the theorem
theorem integral_f_equals_four_thirds :
  ∫ x in (0)..(2), f x = 4/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_equals_four_thirds_l588_58826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_largest_factorial_consecutive_product_l588_58885

/-- 
Given a positive integer n, returns true if n factorial can be expressed as 
the product of n - 4 consecutive positive integers, false otherwise.
-/
def can_express (n : ℕ+) : Prop :=
  ∃ k : ℕ+, Nat.factorial n.val = (k * (k+1) * (k+2) * (k+3))

/-- 
Theorem stating that 1 is the largest positive integer n for which n factorial 
can be expressed as the product of n - 4 consecutive positive integers.
-/
theorem largest_factorial_consecutive_product : 
  (can_express 1) ∧ (∀ m : ℕ+, m > 1 → ¬(can_express m)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_largest_factorial_consecutive_product_l588_58885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_ellipse_to_line_l588_58805

/-- The ellipse defined by the equation x²/16 + y²/4 = 1 -/
def ellipse (x y : ℝ) : Prop :=
  x^2 / 16 + y^2 / 4 = 1

/-- The line defined by the equation x + 2y - √2 = 0 -/
def line (x y : ℝ) : Prop :=
  x + 2*y - Real.sqrt 2 = 0

/-- The distance from a point (x, y) to the line x + 2y - √2 = 0 -/
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  abs (x + 2*y - Real.sqrt 2) / Real.sqrt 5

/-- Theorem: The maximum distance from any point on the ellipse to the line is √10 -/
theorem max_distance_ellipse_to_line :
  ∀ x y : ℝ, ellipse x y → distance_to_line x y ≤ Real.sqrt 10 ∧
  ∃ x₀ y₀ : ℝ, ellipse x₀ y₀ ∧ distance_to_line x₀ y₀ = Real.sqrt 10 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_ellipse_to_line_l588_58805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_girls_fraction_is_half_l588_58870

/-- Represents a school with a given number of students and boy-to-girl ratio -/
structure School where
  totalStudents : ℕ
  boyRatio : ℕ
  girlRatio : ℕ

/-- Calculates the number of girls in a school -/
def girlsInSchool (s : School) : ℕ :=
  s.totalStudents * s.girlRatio / (s.boyRatio + s.girlRatio)

/-- The combined event with two schools -/
def combinedEvent (s1 s2 : School) : ℕ × ℕ :=
  (s1.totalStudents + s2.totalStudents, girlsInSchool s1 + girlsInSchool s2)

theorem girls_fraction_is_half (maplewood brookside : School)
    (h1 : maplewood.totalStudents = 300)
    (h2 : maplewood.boyRatio = 3)
    (h3 : maplewood.girlRatio = 2)
    (h4 : brookside.totalStudents = 240)
    (h5 : brookside.boyRatio = 3)
    (h6 : brookside.girlRatio = 5) :
    let (total, girls) := combinedEvent maplewood brookside
    2 * girls = total := by
  sorry

#eval girlsInSchool { totalStudents := 300, boyRatio := 3, girlRatio := 2 }
#eval girlsInSchool { totalStudents := 240, boyRatio := 3, girlRatio := 5 }
#eval combinedEvent 
  { totalStudents := 300, boyRatio := 3, girlRatio := 2 }
  { totalStudents := 240, boyRatio := 3, girlRatio := 5 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_girls_fraction_is_half_l588_58870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_X_value_l588_58889

noncomputable def F (X : ℝ) : Finset ℝ := {-4, -1, 0, X, 9}

noncomputable def original_mean (X : ℝ) : ℝ := ((-4) + (-1) + 0 + X + 9) / 5

noncomputable def new_mean (X : ℝ) : ℝ := (2 + 3 + 0 + X + 9) / 5

theorem unique_X_value :
  ∃! X : ℝ, 
    (∀ x ∈ F X, x ≥ -4) ∧
    (∀ x ∈ F X, x ≤ 9) ∧
    new_mean X ≥ 2 * original_mean X ∧
    new_mean X < 2 * original_mean X + 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_X_value_l588_58889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_waiting_time_l588_58887

/-- Represents the duration of the traffic light cycle in minutes -/
noncomputable def cycle_duration : ℝ := 3

/-- Represents the duration of the green light in minutes -/
noncomputable def green_duration : ℝ := 1

/-- Represents the duration of the red light in minutes -/
noncomputable def red_duration : ℝ := cycle_duration - green_duration

/-- Represents the probability of arriving during the green light -/
noncomputable def prob_green : ℝ := green_duration / cycle_duration

/-- Represents the probability of arriving during the red light -/
noncomputable def prob_red : ℝ := red_duration / cycle_duration

/-- Represents the expected waiting time if arriving during the green light -/
noncomputable def expected_wait_green : ℝ := 0

/-- Represents the expected waiting time if arriving during the red light -/
noncomputable def expected_wait_red : ℝ := red_duration / 2

/-- Theorem stating that the expected waiting time is 2/3 minutes -/
theorem expected_waiting_time :
  prob_green * expected_wait_green + prob_red * expected_wait_red = 2/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_waiting_time_l588_58887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orangeade_price_problem_l588_58872

/-- Represents the price of orangeade per glass on a given day. -/
structure OrangeadePrice where
  price : ℝ
  day : Nat

/-- Represents the amount of liquid in liters. -/
@[ext] structure Liquid where
  amount : ℝ

/-- Represents the revenue from selling orangeade. -/
def revenue (price : ℝ) (volume : ℝ) : ℝ := price * volume

/-- The orangeade scenario on two consecutive days. -/
theorem orangeade_price_problem (orange_juice : Liquid) (water_day1 : Liquid) (water_day2 : Liquid)
    (price_day1 price_day2 : OrangeadePrice) (glasses_per_liter : ℝ) :
    orange_juice.amount = water_day1.amount →
    water_day2.amount = 2 * water_day1.amount →
    price_day2.price = 0.5466666666666666 →
    price_day2.day = 2 →
    revenue price_day1.price ((orange_juice.amount + water_day1.amount) * glasses_per_liter) =
      revenue price_day2.price ((orange_juice.amount + water_day2.amount) * glasses_per_liter) →
    price_day1.price = 0.82 ∧ price_day1.day = 1 :=
  by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_orangeade_price_problem_l588_58872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_magnitude_proof_l588_58817

theorem complex_magnitude_proof : Complex.abs (2/3 - (5/4)*Complex.I) = 17/12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_magnitude_proof_l588_58817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_diagonals_property_not_in_rectangle_l588_58865

structure Quadrilateral where
  sides : Fin 4 → ℝ
  angles : Fin 4 → ℝ
  diagonals : Fin 2 → ℝ

structure Rhombus extends Quadrilateral where
  all_sides_equal : ∀ i j : Fin 4, sides i = sides j
  opposite_sides_parallel : True  -- Simplified for this example
  opposite_angles_equal : angles 0 = angles 2 ∧ angles 1 = angles 3
  diagonals_perpendicular : Prop  -- Changed to Prop
  diagonals_bisect : Prop  -- Changed to Prop

structure Rectangle extends Quadrilateral where
  opposite_sides_equal : sides 0 = sides 2 ∧ sides 1 = sides 3
  opposite_sides_parallel : True  -- Simplified for this example
  all_angles_right : ∀ i : Fin 4, angles i = 90
  diagonals_equal : diagonals 0 = diagonals 1
  diagonals_bisect : Prop  -- Changed to Prop

theorem rhombus_diagonals_property_not_in_rectangle :
  ∃ (r : Rhombus), ∀ (rect : Rectangle),
    (r.diagonals_perpendicular ∧ r.diagonals_bisect) ∧
    ¬(∃ (h : Prop), h ∧ rect.diagonals_bisect) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_diagonals_property_not_in_rectangle_l588_58865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_operation_schemes_l588_58842

theorem cube_operation_schemes :
  ∀ (n : ℕ) (a : ℕ), n ∈ ({7, 11, 13, 17} : Set ℕ) → a ∈ ({2, 3, 4} : Set ℕ) →
    (a^3 : ℤ) % n = 
      match n, a with
      | 7,  2 => 1
      | 7,  3 => 6
      | 7,  4 => 1
      | 11, 2 => 8
      | 11, 3 => 5
      | 11, 4 => 9
      | 13, 2 => 8
      | 13, 3 => 1
      | 13, 4 => 12
      | 17, 2 => 8
      | 17, 3 => 10
      | 17, 4 => 13
      | _, _ => 0  -- This case should never occur due to the conditions
      := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_operation_schemes_l588_58842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l588_58895

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (-15 * x^2 + 14 * x + 8)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y : ℝ, f x = y} = {x : ℝ | -2/5 ≤ x ∧ x ≤ 4/3} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l588_58895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_foci_distance_product_l588_58869

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 6 + y^2 / 2 = 1

-- Define the line
def line (k m x : ℝ) : ℝ := k * x + m

-- Define the foci
def F1 : ℝ × ℝ := (-2, 0)
def F2 : ℝ × ℝ := (2, 0)

-- Define the tangency condition
def is_tangent (k m : ℝ) : Prop := 
  ∃ x y, is_on_ellipse x y ∧ y = line k m x ∧ 
  ∀ x' y', is_on_ellipse x' y' → y' ≠ line k m x' ∨ (x' = x ∧ y' = y)

-- Define the distance from a point to a line
noncomputable def distance_point_to_line (x y k m : ℝ) : ℝ :=
  abs (k * x - y + m) / Real.sqrt (k^2 + 1)

-- State the theorem
theorem foci_distance_product (k m : ℝ) 
  (h : is_tangent k m) : 
  distance_point_to_line F1.1 F1.2 k m * distance_point_to_line F2.1 F2.2 k m = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_foci_distance_product_l588_58869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_of_barium_hydroxide_l588_58841

/-- The molar mass of Barium hydroxide in g/mol -/
def molar_mass_Ba_OH_2 : ℝ := 171.35

/-- The weight of Barium hydroxide in grams -/
def weight_Ba_OH_2 : ℝ := 513

/-- The number of moles of Barium hydroxide -/
def moles_Ba_OH_2 : ℝ := 3

/-- Tolerance for approximation -/
def tolerance : ℝ := 0.1

theorem weight_of_barium_hydroxide :
  |weight_Ba_OH_2 - moles_Ba_OH_2 * molar_mass_Ba_OH_2| ≤ tolerance := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_of_barium_hydroxide_l588_58841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sarah_expected_red_area_l588_58811

/-- Represents a circle in the dart game -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a player in the dart game -/
inductive Player
  | Sarah
  | Hagar

/-- The dart game as described in the problem -/
structure DartGame where
  initial_circle : Circle
  current_player : Player
  colored_area : (ℝ × ℝ) → Bool

/-- The expected value of the area colored by a player -/
noncomputable def expected_colored_area (player : Player) (game : DartGame) : ℝ := sorry

theorem sarah_expected_red_area :
  ∀ (game : DartGame),
    game.initial_circle.radius = 1 →
    game.current_player = Player.Sarah →
    expected_colored_area Player.Sarah game = (6 * Real.pi ^ 2) / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sarah_expected_red_area_l588_58811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l588_58823

def sequenceProperty (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, a n = 3 * S n - 2

theorem sequence_general_term (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h : sequenceProperty a S) : 
  ∀ n : ℕ, a n = (-1/2 : ℝ)^(n - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l588_58823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_of_negative_x_minus_one_l588_58812

noncomputable def slope_angle (m : ℝ) : ℝ := Real.arctan m

def line_equation (x y : ℝ) : Prop := y = -x - 1

def slope_of_line (m : ℝ) : ℝ := m

theorem slope_angle_of_negative_x_minus_one :
  slope_angle (slope_of_line (-1)) = 3 * π / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_of_negative_x_minus_one_l588_58812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_least_60_cents_l588_58879

/-- The number of pennies in the box -/
def num_pennies : ℕ := 3

/-- The number of nickels in the box -/
def num_nickels : ℕ := 5

/-- The number of dimes in the box -/
def num_dimes : ℕ := 7

/-- The total number of coins in the box -/
def total_coins : ℕ := num_pennies + num_nickels + num_dimes

/-- The number of coins drawn -/
def coins_drawn : ℕ := 7

/-- The probability of drawing coins worth at least 60 cents -/
theorem probability_at_least_60_cents : 
  (Nat.choose total_coins coins_drawn : ℚ)⁻¹ * 
  (Nat.choose num_nickels 2 * Nat.choose num_dimes 5 + 
   Nat.choose num_nickels 1 * Nat.choose num_dimes 6 + 
   Nat.choose num_dimes 7 : ℚ) = 246 / 6435 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_least_60_cents_l588_58879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_decrease_l588_58848

structure Car where
  initialSpeed : ℝ
  finalSpeed : ℝ

noncomputable def totalTime : ℝ := 40 / 60 -- 40 minutes in hours

theorem car_speed_decrease (carA carB : Car) : 
  carA.initialSpeed = carB.initialSpeed + 2.5 →
  carA.finalSpeed = carB.finalSpeed + 0.5 →
  (carA.initialSpeed * (10 / 60) + carA.finalSpeed * (30 / 60)) = 
  (carB.initialSpeed * (15 / 60) + carB.finalSpeed * (25 / 60)) →
  carA.initialSpeed - carA.finalSpeed = 10 := by
  sorry

#check car_speed_decrease

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_decrease_l588_58848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stocking_cost_prove_stocking_cost_l588_58830

theorem stocking_cost (num_grandchildren num_children : ℕ) 
  (stockings_per_person : ℕ) (stocking_price : ℚ) (discount_rate : ℚ) 
  (monogram_price : ℚ) : Prop :=
  let total_stockings := (num_grandchildren + num_children) * stockings_per_person
  let original_price := total_stockings * stocking_price
  let discounted_price := original_price * (1 - discount_rate)
  let monogram_cost := total_stockings * monogram_price
  discounted_price + monogram_cost = 1035 ∧
  num_grandchildren = 5 ∧
  num_children = 4 ∧
  stockings_per_person = 5 ∧
  stocking_price = 20 ∧
  discount_rate = 1/10 ∧
  monogram_price = 5

theorem prove_stocking_cost : stocking_cost 5 4 5 20 (1/10) 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stocking_cost_prove_stocking_cost_l588_58830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_line_ellipse_l588_58859

/-- Line l in rectangular coordinates -/
def line_l (x y : ℝ) : Prop := x + 2*y + 3*Real.sqrt 2 = 0

/-- Ellipse C in rectangular coordinates -/
def ellipse_C (x y : ℝ) : Prop := x^2/4 + y^2 = 1

/-- Distance between two points -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

/-- Minimum distance between line l and ellipse C -/
theorem min_distance_line_ellipse :
  ∃ (min_dist : ℝ), min_dist = Real.sqrt 10 / 5 ∧
    ∀ (x1 y1 x2 y2 : ℝ), line_l x1 y1 → ellipse_C x2 y2 →
      distance x1 y1 x2 y2 ≥ min_dist := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_line_ellipse_l588_58859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_problem_l588_58891

/-- The sum of the first n terms of a geometric sequence -/
noncomputable def geometric_sum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sequence_sum_problem (n : ℕ) :
  geometric_sum 1 (1/2) n = 31/16 → n = 5 := by
  intro h
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_problem_l588_58891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polyhedron_vertex_degree_l588_58813

-- Define a convex polyhedron
structure ConvexPolyhedron where
  vertices : Set (ℕ × ℕ × ℕ)
  edges : Set (ℕ × ℕ)
  faces : Set (Set (ℕ × ℕ × ℕ))
  convex : Bool

-- Define a coloring of vertices
def Coloring (p : ConvexPolyhedron) := (ℕ × ℕ × ℕ) → Bool

-- Define a function to get the degree of a vertex
def degree (p : ConvexPolyhedron) (v : ℕ × ℕ × ℕ) : ℕ := sorry

-- Define a predicate for a valid coloring (one face red, rest blue)
def validColoring (p : ConvexPolyhedron) (c : Coloring p) : Prop := sorry

-- Theorem statement
theorem polyhedron_vertex_degree 
  (p : ConvexPolyhedron) 
  (c : Coloring p) 
  (h : validColoring p c) : 
  (∃ v ∈ p.vertices, ¬(c v) ∧ degree p v ≤ 5) ∨ 
  (∃ v ∈ p.vertices, c v ∧ degree p v = 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polyhedron_vertex_degree_l588_58813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_walkway_middle_position_l588_58835

/-- Represents the position of a person on or beside the walkway at time t -/
noncomputable def position (start_time speed : ℝ) (t : ℝ) : ℝ :=
  if t ≥ start_time then
    (t - start_time) * speed
  else
    0

/-- The problem statement -/
theorem walkway_middle_position :
  let walkway_speed : ℝ := 6
  let al_speed : ℝ := walkway_speed
  let bob_speed : ℝ := walkway_speed + 4
  let cy_speed : ℝ := 8
  let al_start : ℝ := 0
  let bob_start : ℝ := 2
  let cy_start : ℝ := 4
  ∃ t : ℝ, t > cy_start ∧
    let al_pos := position al_start al_speed t
    let bob_pos := position bob_start bob_speed t
    let cy_pos := position cy_start cy_speed t
    (al_pos = (bob_pos + cy_pos) / 2 ∨
     bob_pos = (al_pos + cy_pos) / 2 ∨
     cy_pos = (al_pos + bob_pos) / 2) ∧
    al_pos = 52 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_walkway_middle_position_l588_58835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_channel_area_l588_58800

/-- Calculates the area of a trapezoidal cross-section of a water channel. -/
noncomputable def trapezoidalChannelArea (topWidth bottomWidth depth : ℝ) : ℝ :=
  (1 / 2) * (topWidth + bottomWidth) * depth

/-- Theorem: The area of a trapezoidal water channel with top width 14 meters,
    bottom width 8 meters, and depth 80 meters is 880 square meters. -/
theorem water_channel_area :
  trapezoidalChannelArea 14 8 80 = 880 := by
  -- Unfold the definition of trapezoidalChannelArea
  unfold trapezoidalChannelArea
  -- Simplify the arithmetic
  simp [mul_add, mul_comm, mul_assoc]
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_channel_area_l588_58800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_y_coordinate_comparison_l588_58832

/-- Prove that for a downward-opening parabola y = a(x+2)² + 3 where a < 0,
    the y-coordinate of the point (-1, y₁) is greater than
    the y-coordinate of the point (2, y₂) -/
theorem parabola_y_coordinate_comparison
  (a : ℝ) (y₁ y₂ : ℝ)
  (h_a : a < 0)
  (h_y₁ : y₁ = a * (-1 + 2)^2 + 3)
  (h_y₂ : y₂ = a * (2 + 2)^2 + 3) :
  y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_y_coordinate_comparison_l588_58832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_16380_l588_58810

theorem number_of_factors_16380 : ∃ (n : ℕ), n = 72 ∧ n = (Finset.filter (λ x : ℕ ↦ 16380 % x = 0) (Finset.range (16380 + 1))).card := by
  let factors := Finset.filter (λ x : ℕ ↦ 16380 % x = 0) (Finset.range (16380 + 1))
  let n := factors.card
  use n
  apply And.intro
  · sorry -- Proof that n = 72
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_16380_l588_58810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_x_value_l588_58854

-- Define the condition from the problem
def satisfies_equation (x y : ℝ) : Prop :=
  x^2 + y^2 = 18*x + 50*y + 56

-- Define the minimum value of x
noncomputable def min_x : ℝ := 9 - Real.sqrt 762

-- Theorem statement
theorem minimum_x_value (x y : ℝ) (h : satisfies_equation x y) :
  x ≥ min_x := by
  sorry

#check minimum_x_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_x_value_l588_58854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_line_and_estimate_correct_l588_58856

-- Define the given data
def x_bar : ℚ := 69/100
def y_bar : ℚ := 28/100
def sum_xy : ℚ := 19951/10000
def sum_x_squared : ℚ := 49404/10000
def k : ℕ := 10

-- Define the regression coefficients
noncomputable def b_hat : ℚ := (sum_xy - k * x_bar * y_bar) / (sum_x_squared - k * x_bar^2)
noncomputable def a_hat : ℚ := y_bar - b_hat * x_bar

-- Define the regression line function
def regression_line (x : ℚ) : ℚ := 35/100 * x + 4/100

-- Define the estimated empty shell rate at 40% humidity
def estimated_rate_at_40 : ℚ := regression_line (40/100)

-- Theorem statement
theorem regression_line_and_estimate_correct :
  (∀ x, regression_line x = 35/100 * x + 4/100) ∧
  estimated_rate_at_40 = 18/100 := by
  sorry

#eval regression_line (40/100)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_line_and_estimate_correct_l588_58856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_reciprocal_sum_l588_58882

-- Define the curve C
noncomputable def curve_C (x y : ℝ) : Prop := y^2 - x^2 = 4

-- Define the line l
noncomputable def line_l (t x y : ℝ) : Prop :=
  x = t / Real.sqrt 5 ∧ y = Real.sqrt 5 + 2 * t / Real.sqrt 5

-- Define point A
noncomputable def point_A : ℝ × ℝ := (0, Real.sqrt 5)

-- Define the theorem
theorem intersection_reciprocal_sum :
  ∃ (M N : ℝ × ℝ),
    (curve_C M.1 M.2) ∧
    (curve_C N.1 N.2) ∧
    (∃ t₁ t₂ : ℝ, line_l t₁ M.1 M.2 ∧ line_l t₂ N.1 N.2) ∧
    (1 / Real.sqrt ((M.1 - point_A.1)^2 + (M.2 - point_A.2)^2) +
     1 / Real.sqrt ((N.1 - point_A.1)^2 + (N.2 - point_A.2)^2) = 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_reciprocal_sum_l588_58882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_range_l588_58847

noncomputable def f (x : ℝ) := x + Real.cos x - Real.sqrt 3 * Real.sin x

theorem tangent_slope_range :
  ∀ x : ℝ, -1 ≤ (deriv f) x ∧ (deriv f) x ≤ 3 := by
  intro x
  -- The proof steps would go here
  sorry

#check tangent_slope_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_range_l588_58847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_solutions_in_interval_l588_58876

noncomputable def g (n : ℕ) (x : ℝ) : ℝ := Real.sin x ^ n + Real.cos x ^ n

theorem five_solutions_in_interval :
  ∃! (S : Finset ℝ), S.card = 5 ∧
    (∀ x ∈ S, 0 ≤ x ∧ x ≤ 2 * Real.pi) ∧
    (∀ x ∈ S, 4 * g 6 x - 3 * g 8 x = g 2 x) ∧
    (∀ y, 0 ≤ y ∧ y ≤ 2 * Real.pi → 4 * g 6 y - 3 * g 8 y = g 2 y → y ∈ S) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_solutions_in_interval_l588_58876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l588_58898

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (3 * x^2 - 2) / (x^2 + 1)

-- State the theorem
theorem range_of_f :
  {y : ℝ | ∃ x ≥ 0, f x = y} = Set.Icc (-2 : ℝ) 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l588_58898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l588_58844

noncomputable def f (x : ℝ) := Real.log x / Real.log 3 - 1 / x

theorem zero_in_interval :
  ∃ c : ℝ, c > 1 ∧ c < 2 ∧ f c = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l588_58844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_average_speed_l588_58894

noncomputable def average_speed (distance_1 : ℝ) (distance_2 : ℝ) (time_1 : ℝ) (time_2 : ℝ) : ℝ :=
  (distance_1 + distance_2) / (time_1 + time_2)

theorem car_average_speed :
  let distance_1 : ℝ := 65
  let distance_2 : ℝ := 45
  let time_1 : ℝ := 1
  let time_2 : ℝ := 1
  average_speed distance_1 distance_2 time_1 time_2 = 55 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_average_speed_l588_58894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_weight_theorem_l588_58896

theorem candy_weight_theorem (n : ℕ) (weights : List ℕ) 
  (h1 : n = 14)
  (h2 : weights.length = n)
  (h3 : (weights.sum : ℝ) / n ≥ 90.15)
  (h4 : (weights.sum : ℝ) / n < 90.25) :
  weights.sum = 1263 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_weight_theorem_l588_58896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jessica_bike_ride_time_jessica_bike_ride_proof_l588_58818

/-- The time taken for Jessica to ride her bike along semicircular paths on a highway -/
theorem jessica_bike_ride_time 
  (highway_length : ℝ) 
  (highway_width : ℝ) 
  (bike_speed : ℝ) 
  (h1 : highway_length = 2) -- 2 miles long
  (h2 : highway_width = 60 / 5280) -- 60 feet converted to miles
  (h3 : bike_speed = 6) -- 6 miles per hour
  : ℝ :=
  Real.pi / 6

theorem jessica_bike_ride_proof 
  (highway_length : ℝ) 
  (highway_width : ℝ) 
  (bike_speed : ℝ) 
  (h1 : highway_length = 2)
  (h2 : highway_width = 60 / 5280)
  (h3 : bike_speed = 6)
  : jessica_bike_ride_time highway_length highway_width bike_speed h1 h2 h3 = Real.pi / 6 := by
  sorry

#check jessica_bike_ride_time
#check jessica_bike_ride_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jessica_bike_ride_time_jessica_bike_ride_proof_l588_58818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_identity_l588_58839

open Matrix

variable {n : Type*} [DecidableEq n] [Fintype n]

theorem matrix_identity (B : Matrix n n ℝ) (h_inv : Invertible B) 
  (h_eq : (B - 3 • (1 : Matrix n n ℝ)) * (B - 5 • (1 : Matrix n n ℝ)) = 0) : 
  B + 10 • B⁻¹ = (40/3) • (1 : Matrix n n ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_identity_l588_58839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dividing_line_theorem_l588_58820

/-- Triangle ABC with vertices A(1,3), B(1,0), and C(10,0) -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Area of a triangle given its base and height -/
noncomputable def triangleArea (base height : ℝ) : ℝ := (1/2) * base * height

/-- Area of the left region of the triangle when divided by a vertical line -/
noncomputable def leftArea (t : Triangle) (a : ℝ) : ℝ :=
  triangleArea (a - t.A.1) t.A.2

/-- Area of the right region of the triangle when divided by a vertical line -/
noncomputable def rightArea (t : Triangle) (a : ℝ) : ℝ :=
  triangleArea (t.C.1 - a) t.A.2

/-- The theorem to be proved -/
theorem dividing_line_theorem (t : Triangle) (a : ℝ) :
  t.A = (1, 3) ∧ t.B = (1, 0) ∧ t.C = (10, 0) ∧ a = 7.75 →
  leftArea t a = 3 * rightArea t a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dividing_line_theorem_l588_58820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_theorem_l588_58867

-- Define the circle
def circle_radius : ℝ := 5

-- Define the angle of intersection
def intersection_angle : ℝ := 60

-- Define the area of an equilateral triangle with side length s
noncomputable def equilateral_triangle_area (s : ℝ) : ℝ := (Real.sqrt 3 / 4) * s^2

-- Define the area of a sector with central angle θ (in degrees) in a circle of radius r
noncomputable def sector_area (r : ℝ) (θ : ℝ) : ℝ := (θ / 360) * Real.pi * r^2

theorem shaded_area_theorem :
  2 * (equilateral_triangle_area circle_radius) + 2 * (sector_area circle_radius intersection_angle) =
  (25 * Real.sqrt 3) / 2 + (25 * Real.pi) / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_theorem_l588_58867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_V_is_not_polynomial_l588_58892

-- Define the function V(X)
noncomputable def V (X : ℝ) : ℝ := 1 / (X^2 + 1)

-- Theorem stating that V is not a polynomial
theorem V_is_not_polynomial :
  ¬ ∃ (P : Polynomial ℝ), ∀ (X : ℝ), V X = P.eval X := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_V_is_not_polynomial_l588_58892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_in_fourth_quadrant_l588_58853

-- Define the point P as noncomputable due to the use of Real.sqrt
noncomputable def P : ℝ × ℝ := (Real.sqrt 2022, -Real.sqrt 2023)

-- Define the conditions
axiom sqrt_2022_pos : Real.sqrt 2022 > 0
axiom neg_sqrt_2023_neg : -Real.sqrt 2023 < 0

-- Define what it means for a point to be in the fourth quadrant
def in_fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

-- State the theorem
theorem P_in_fourth_quadrant : in_fourth_quadrant P := by
  unfold in_fourth_quadrant P
  constructor
  · exact sqrt_2022_pos
  · exact neg_sqrt_2023_neg


end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_in_fourth_quadrant_l588_58853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_courtyard_length_is_30_l588_58809

/-- Represents the dimensions of a paving stone -/
structure PavingStone where
  length : ℝ
  width : ℝ

/-- Represents a rectangular courtyard -/
structure Courtyard where
  length : ℝ
  width : ℝ

/-- Calculates the area of a paving stone -/
def PavingStone.area (stone : PavingStone) : ℝ := stone.length * stone.width

/-- Calculates the area of a courtyard -/
def Courtyard.area (yard : Courtyard) : ℝ := yard.length * yard.width

/-- Theorem: The length of the courtyard is 30 meters -/
theorem courtyard_length_is_30 (stone : PavingStone) (yard : Courtyard) 
    (h1 : stone.length = 2)
    (h2 : stone.width = 1)
    (h3 : yard.width = 16)
    (h4 : yard.area = 240 * stone.area) : 
  yard.length = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_courtyard_length_is_30_l588_58809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_surface_area_circumradius_ratio_l588_58858

/-- A right tetrahedron with perpendicular edges -/
structure RightTetrahedron where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c

/-- Surface area of the right tetrahedron -/
noncomputable def surface_area (t : RightTetrahedron) : ℝ :=
  2 * (t.a * t.b + t.b * t.c + t.c * t.a)

/-- Radius of the circumscribed sphere -/
noncomputable def circumradius (t : RightTetrahedron) : ℝ :=
  (1 / 2) * Real.sqrt (t.a^2 + t.b^2 + t.c^2)

/-- Theorem: The maximum value of S/R² for a right tetrahedron -/
theorem max_surface_area_circumradius_ratio :
    ∀ t : RightTetrahedron,
      (surface_area t / (circumradius t)^2) ≤ (2/3) * (3 + Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_surface_area_circumradius_ratio_l588_58858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_at_90_deg_is_12_5_l588_58863

/-- Represents an elliptical orbit -/
structure EllipticalOrbit where
  perigee : ℝ
  apogee : ℝ

/-- Calculates the distance from the focus to a point on the orbit at 90° true anomaly -/
noncomputable def distance_at_90_deg (orbit : EllipticalOrbit) : ℝ :=
  let semi_major_axis := (orbit.apogee + orbit.perigee) / 2
  let c := semi_major_axis - orbit.perigee
  let semi_minor_axis := Real.sqrt (semi_major_axis^2 - c^2)
  Real.sqrt (semi_minor_axis^2 + c^2)

/-- Theorem stating that for the given orbit, the distance at 90° true anomaly is 12.5 AU -/
theorem distance_at_90_deg_is_12_5 :
  let orbit := EllipticalOrbit.mk 5 20
  distance_at_90_deg orbit = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_at_90_deg_is_12_5_l588_58863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_three_no_three_digit_l588_58871

/-- A function that checks if a number contains the digit 3 -/
def containsThree (n : Nat) : Bool :=
  sorry

/-- A function that counts numbers in a range satisfying certain conditions -/
def countNumbers (start : Nat) (stop : Nat) (divisor : Nat) : Nat :=
  sorry

theorem divisible_by_three_no_three_digit : 
  countNumbers 1 100 3 = 26 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_three_no_three_digit_l588_58871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frank_finishes_first_l588_58846

-- Define the areas of the gardens
noncomputable def franks_garden_area : ℝ := 1
noncomputable def emilys_garden_area : ℝ := 4 * franks_garden_area
noncomputable def davids_garden_area : ℝ := 3 * emilys_garden_area

-- Define the mowing rates
noncomputable def franks_mower_rate : ℝ := 1
noncomputable def emilys_mower_rate : ℝ := 4 * franks_mower_rate
noncomputable def davids_mower_rate : ℝ := 2 * franks_mower_rate

-- Define Emily's mower efficiency
noncomputable def emilys_mower_efficiency : ℝ := 0.75

-- Calculate mowing times
noncomputable def franks_mowing_time : ℝ := franks_garden_area / franks_mower_rate
noncomputable def emilys_mowing_time : ℝ := emilys_garden_area / (emilys_mower_rate * emilys_mower_efficiency)
noncomputable def davids_mowing_time : ℝ := davids_garden_area / davids_mower_rate

-- Theorem statement
theorem frank_finishes_first :
  franks_mowing_time < emilys_mowing_time ∧ franks_mowing_time < davids_mowing_time := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frank_finishes_first_l588_58846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_safe_choice_l588_58814

-- Define the types of sisters
inductive SisterType
| Knight
| Liar
| Normal

-- Define the sisters
structure Sister :=
  (name : Char)
  (type : SisterType)
  (isWerewolf : Bool)

-- Define the question and answer
def question (b c : Sister) : Bool := b.name < c.name

def answer (a b c : Sister) : Bool :=
  match a.type with
  | SisterType.Knight => question b c
  | SisterType.Liar => !(question b c)
  | SisterType.Normal => true  -- Can be either true or false

-- Define the theorem
theorem safe_choice (a b c : Sister) :
  (a.name = 'A' ∧ b.name = 'B' ∧ c.name = 'C') →
  (a.type ≠ b.type ∧ b.type ≠ c.type ∧ c.type ≠ a.type) →
  (a.isWerewolf ∨ b.isWerewolf ∨ c.isWerewolf) →
  (a.isWerewolf → a.type = SisterType.Normal) →
  (b.isWerewolf → b.type = SisterType.Normal) →
  (c.isWerewolf → c.type = SisterType.Normal) →
  (answer a b c → ¬b.isWerewolf) ∧
  (¬(answer a b c) → ¬c.isWerewolf) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_safe_choice_l588_58814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_two_zeros_properties_l588_58855

open Real

/-- The function f(x) = ln x - ax + 1 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log x - a * x + 1

/-- Theorem stating the properties of the function f when it has two distinct zeros -/
theorem f_two_zeros_properties (a : ℝ) (x₁ x₂ : ℝ) 
  (h_zeros : f a x₁ = 0 ∧ f a x₂ = 0)
  (h_distinct : x₁ < x₂) :
  a ∈ Set.Ioo 0 1 ∧ x₁ + x₂ > 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_two_zeros_properties_l588_58855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_is_three_l588_58808

/-- Sum of the first n terms of an arithmetic sequence with first term a and common difference 5 -/
noncomputable def T (n : ℕ+) (a : ℝ) : ℝ := n * (2 * a + (n - 1) * 5) / 2

/-- The theorem stating that if the ratio of T_{4n} to T_n is constant, then the first term is 3 -/
theorem first_term_is_three (h : ∃ k : ℝ, ∀ n : ℕ+, T (4 * n) 3 / T n 3 = k) : 
  ∃ a : ℝ, ∀ n : ℕ+, T (4 * n) a / T n a = T (4 * n) 3 / T n 3 → a = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_is_three_l588_58808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_terminal_side_l588_58864

/-- Given sin α = -3/5 and cos α = 4/5, prove that (4, -3) lies on the terminal side of angle α -/
theorem point_on_terminal_side (α : ℝ) (h1 : Real.sin α = -3/5) (h2 : Real.cos α = 4/5) :
  ∃ (k : ℝ), k > 0 ∧ k * 4 = 4 ∧ k * (-3) = -3 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_terminal_side_l588_58864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_is_45_degrees_l588_58886

-- Define a triangle ABC
structure Triangle where
  A : ℝ  -- Angle A
  B : ℝ  -- Angle B
  C : ℝ  -- Angle C
  a : ℝ  -- Side opposite to A
  b : ℝ  -- Side opposite to B
  c : ℝ  -- Side opposite to C

-- Define the problem conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a = 2 * Real.sqrt 3 ∧
  t.c = 2 * Real.sqrt 2 ∧
  1 + (Real.tan t.A / Real.tan t.B) = 2 * t.c / t.b

-- Theorem statement
theorem angle_C_is_45_degrees (t : Triangle) 
  (h : triangle_conditions t) : t.C = Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_is_45_degrees_l588_58886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_l588_58857

noncomputable def f (A B C : ℤ) (x : ℝ) : ℝ := x^2 / (A * x^2 + B * x + C)

theorem sum_of_coefficients (A B C : ℤ) :
  (∀ x > 4, f A B C x > 0.4) →
  (A * (-2)^2 + B * (-2) + C = 0) →
  (A * 3^2 + B * 3 + C = 0) →
  (0.4 < (1 : ℝ) / A ∧ (1 : ℝ) / A < 1) →
  A + B + C = -12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_l588_58857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_parabola_and_x_axis_l588_58816

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 1

-- Define the area enclosed by the function and the x-axis
noncomputable def enclosed_area : ℝ := ∫ x in (-1)..1, (0 - f x)

-- Theorem statement
theorem area_enclosed_by_parabola_and_x_axis :
  enclosed_area = 4/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_parabola_and_x_axis_l588_58816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l588_58899

noncomputable def vector_a : Fin 2 → ℝ := ![1, 0]
noncomputable def vector_b : Fin 2 → ℝ := ![-1/2, Real.sqrt 3/2]

theorem angle_between_vectors :
  let dot_product := (vector_a 0) * (vector_b 0) + (vector_a 1) * (vector_b 1)
  let magnitude_a := Real.sqrt ((vector_a 0)^2 + (vector_a 1)^2)
  let magnitude_b := Real.sqrt ((vector_b 0)^2 + (vector_b 1)^2)
  let cos_theta := dot_product / (magnitude_a * magnitude_b)
  Real.arccos cos_theta = 2 * π / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l588_58899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l588_58801

def a : ℝ × ℝ := (1, -1)
def b (l : ℝ) : ℝ × ℝ := (l, 1)

def angle_obtuse (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 < 0

theorem lambda_range (l : ℝ) :
  angle_obtuse a (b l) → l < -1 ∨ (-1 < l ∧ l < 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l588_58801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l588_58873

-- Define the function f(x) with parameter a
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a/2) * Real.sin (2*x) - Real.cos (2*x)

-- State the theorem
theorem function_properties :
  ∃ (a : ℝ),
    (f a (π/8) = 0) ∧
    (a = 2) ∧
    (∀ (x : ℝ), f a (x + π) = f a x) ∧
    (∀ (x : ℝ), f a x ≤ Real.sqrt 2) ∧
    (∃ (y : ℝ), f a y = Real.sqrt 2) := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l588_58873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_complete_circle_parameter_l588_58828

-- Define the polar function r = sin(θ)
noncomputable def r (θ : Real) : Real := Real.sin θ

-- Define what it means for the graph to form a complete circle
def forms_complete_circle (t : Real) : Prop :=
  ∀ θ : Real, ∃ θ' : Real, 0 ≤ θ' ∧ θ' ≤ t ∧ r θ' = r θ

-- State the theorem
theorem smallest_complete_circle_parameter :
  (∃ t : Real, t > 0 ∧ forms_complete_circle t) ∧
  (∀ t : Real, t > 0 ∧ forms_complete_circle t → t ≥ Real.pi) ∧
  forms_complete_circle Real.pi :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_complete_circle_parameter_l588_58828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_path_l588_58878

/-- Represents a square grid -/
structure Grid (n : ℕ) where
  size : n > 0

/-- Represents a path in the grid -/
structure GridPath (n : ℕ) where
  grid : Grid n
  is_closed : Bool
  visits_all_cells : Bool
  no_repeat_diagonals : Bool

/-- Theorem stating the existence of the required path -/
theorem exists_valid_path :
  ∃ (p : GridPath 2018), 
    p.is_closed = true ∧ 
    p.visits_all_cells = true ∧ 
    p.no_repeat_diagonals = true :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_path_l588_58878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candidate_failed_marks_l588_58834

/-- The number of marks by which a candidate fails an exam --/
noncomputable def marksFailed (maxMark : ℝ) (passingPercentage : ℝ) (candidateScore : ℕ) : ℕ :=
  let passingScore := (passingPercentage / 100) * maxMark
  (Int.ceil passingScore - candidateScore).toNat

/-- Theorem stating the number of marks a candidate failed by in the given scenario --/
theorem candidate_failed_marks :
  marksFailed 152.38 42 42 = 22 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_candidate_failed_marks_l588_58834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_increasing_function_l588_58836

open Real MeasureTheory Set

-- Define the function f
def f (a x : ℝ) : ℝ := x^2 + 2*a*x - 1

-- Define the derivative of f with respect to x
def f_deriv (a x : ℝ) : ℝ := 2*x + 2*a

-- Define the interval for a
def a_interval : Set ℝ := Icc (-2) 2

-- Define the condition for f to be increasing
def is_increasing (a : ℝ) : Prop := ∀ x, x ≥ 1 → f_deriv a x ≥ 0

-- State the theorem
theorem probability_increasing_function :
  (volume {a ∈ a_interval | is_increasing a}) / volume a_interval = 3/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_increasing_function_l588_58836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_theorem_l588_58890

/-- The area of the closed region enclosed by f(x) and g(x) -/
noncomputable def enclosed_area (a : ℝ) : ℝ :=
  (2 * Real.pi / a) * Real.sqrt (a^2 + 1)

/-- The function f(x) -/
noncomputable def f (a x : ℝ) : ℝ :=
  a * Real.sin (a * x) + Real.cos (a * x)

/-- The function g(x) -/
noncomputable def g (a : ℝ) : ℝ :=
  Real.sqrt (a^2 + 1)

/-- The theorem stating the area of the enclosed region -/
theorem enclosed_area_theorem (a : ℝ) (ha : a > 0) :
  enclosed_area a = ∫ x in (0)..(2 * Real.pi / a), (g a - f a x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_theorem_l588_58890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_90_root3_l588_58821

/-- Represents a trapezoid ABCD with specific properties -/
structure Trapezoid where
  /-- The length of the base AD -/
  base : ℝ
  /-- The sum of the lengths of the diagonals AC and BD -/
  diagonals_sum : ℝ
  /-- The measure of angle CAD in radians -/
  angle_cad : ℝ
  /-- The ratio of areas of triangles AOD and BOC, where O is the intersection of diagonals -/
  triangles_area_ratio : ℝ

/-- The area of a trapezoid with the given properties -/
noncomputable def trapezoid_area (t : Trapezoid) : ℝ :=
  90 * Real.sqrt 3

/-- Theorem stating that a trapezoid with the given properties has an area of 90√3 -/
theorem trapezoid_area_is_90_root3 (t : Trapezoid) 
    (h1 : t.base = 16) 
    (h2 : t.diagonals_sum = 36) 
    (h3 : t.angle_cad = π / 3)  -- 60° in radians
    (h4 : t.triangles_area_ratio = 4) : 
  trapezoid_area t = 90 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_90_root3_l588_58821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_number_proof_l588_58802

theorem larger_number_proof (a b : ℕ) 
  (h1 : Nat.gcd a b = 20)
  (h2 : Nat.lcm a b = 20 * 3^2 * 17 * 23)
  (h3 : a ≥ b) :
  a = 70380 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_number_proof_l588_58802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_amount_l588_58840

/-- Define the simple interest function -/
noncomputable def simple_interest (P R T : ℝ) : ℝ := (P * R * T) / 100

/-- Theorem statement -/
theorem principal_amount (P R : ℝ) :
  (simple_interest P (R + 5) 10 - simple_interest P R 10 = 600) →
  P = 1200 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_amount_l588_58840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alvin_marble_earnings_l588_58803

/-- The total earnings from selling a set of marbles -/
def total_earnings (total_marbles : ℕ) (white_percent : ℚ) (black_percent : ℚ) 
  (white_price : ℚ) (black_price : ℚ) (color_price : ℚ) : ℚ :=
  let white_marbles := (white_percent * total_marbles) |> Int.floor
  let black_marbles := (black_percent * total_marbles) |> Int.floor
  let color_marbles := total_marbles - white_marbles - black_marbles
  (white_marbles : ℚ) * white_price + 
  (black_marbles : ℚ) * black_price + 
  (color_marbles : ℚ) * color_price

/-- Alvin's marble selling problem -/
theorem alvin_marble_earnings : 
  total_earnings 100 (20/100) (30/100) (5/100) (10/100) (20/100) = 14 := by
  sorry

#eval total_earnings 100 (20/100) (30/100) (5/100) (10/100) (20/100)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alvin_marble_earnings_l588_58803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_quadratic_sum_of_roots_specific_equation_l588_58819

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let eq := λ (x : ℝ) => a * x^2 + b * x + c = 0
  let sum_of_roots := -b / a
  (∃ x y, x ≠ y ∧ eq x ∧ eq y) → sum_of_roots = x + y :=
sorry

theorem sum_of_roots_specific_equation :
  let eq := λ (x : ℝ) => x^2 - 7*x + 2 = 16
  ∃ x y, x ≠ y ∧ eq x ∧ eq y ∧ x + y = 7 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_quadratic_sum_of_roots_specific_equation_l588_58819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_a_range_l588_58877

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then a^x else (3-a)*x + (1/2)*a

def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

theorem f_increasing_a_range (a : ℝ) :
  (is_increasing (f a)) → (a ∈ Set.Icc 2 3 ∧ a ≠ 3) :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_a_range_l588_58877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_M_intersect_N_l588_58825

-- Define the sets M and N
def M : Set ℝ := {x | |x| < 1}
def N : Set ℝ := {y | ∃ x ∈ M, y = 2 * x}

-- State the theorem
theorem complement_of_M_intersect_N :
  (M ∩ N)ᶜ = Set.Iic (-1) ∪ Set.Ici 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_M_intersect_N_l588_58825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_injective_l588_58861

-- Define the τ function (number of divisors)
def τ : ℕ → ℕ := sorry

-- Define the function f
noncomputable def f (n : ℕ) : ℝ := n ^ ((1 : ℝ) / 2 * τ n)

-- State the theorem
theorem f_is_injective : ∀ a b : ℕ, f a = f b → a = b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_injective_l588_58861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_tower_model_height_l588_58849

/-- Represents the height of a water tower model given the original tower's specifications and the model's volume -/
noncomputable def modelHeight (originalHeight : ℝ) (originalVolume : ℝ) (modelVolume : ℝ) : ℝ :=
  originalHeight * (modelVolume / originalVolume) ^ (1/3)

/-- Theorem stating that a model of a specific water tower has the correct height -/
theorem water_tower_model_height :
  let originalHeight : ℝ := 80
  let originalVolume : ℝ := 200000
  let modelVolume : ℝ := 0.2
  modelHeight originalHeight originalVolume modelVolume = 0.8 := by
  sorry

#check water_tower_model_height

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_tower_model_height_l588_58849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locust_population_doubling_time_l588_58888

/-- Represents the growth rate of the locust population per hour -/
noncomputable def r : ℝ := Real.log (128000 / 1000) / 10

/-- Represents the initial population of locusts -/
def P0 : ℝ := 1000

/-- The population growth function -/
noncomputable def P (t : ℝ) : ℝ := P0 * Real.exp (r * t)

/-- The doubling time of the population -/
noncomputable def T : ℝ := Real.log 2 / r

theorem locust_population_doubling_time :
  1.43 < T ∧ T < 2 ∧ P 10 > 128000 ∧ P T = 2 * P0 := by
  sorry

#check locust_population_doubling_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locust_population_doubling_time_l588_58888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l588_58838

/-- Represents the sum of the first n terms of a geometric sequence -/
noncomputable def geometricSum (a1 : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a1 * (1 - q^n) / (1 - q)

/-- Theorem: In a positive geometric sequence with a_1 = 1, if S_5 = 5S_3 - 4, then S_4 = 15 -/
theorem geometric_sequence_sum (q : ℝ) (h_pos : q > 0) :
  let S := geometricSum 1 q
  (S 5 = 5 * S 3 - 4) → S 4 = 15 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l588_58838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_yz_intersection_radius_l588_58852

-- Define the sphere
noncomputable def sphere_center : ℝ × ℝ × ℝ := (3, -2, -10)
noncomputable def sphere_radius : ℝ := Real.sqrt 104

-- Define the xy-plane intersection
def xy_intersection_center : ℝ × ℝ × ℝ := (3, -2, 0)
def xy_intersection_radius : ℝ := 2

-- Define the yz-plane intersection
def yz_intersection_center : ℝ × ℝ × ℝ := (0, -2, -10)

-- Theorem statement
theorem yz_intersection_radius : 
  let r := Real.sqrt (sphere_radius^2 - 3^2)
  r = Real.sqrt 95 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_yz_intersection_radius_l588_58852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_count_l588_58884

-- Define the circle equation
def my_circle (x y : ℝ) : Prop := x^2 + y^2 = 16

-- Define the vertical line equation
def my_vertical_line (x : ℝ) : Prop := x = 3

-- Theorem statement
theorem intersection_points_count :
  ∃ (p q : ℝ × ℝ), p ≠ q ∧
  (∀ (r : ℝ × ℝ), (my_circle r.1 r.2 ∧ my_vertical_line r.1) → (r = p ∨ r = q)) :=
by sorry

#check intersection_points_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_count_l588_58884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_f_l588_58862

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 2*x) / (-1)

-- State the theorem
theorem monotonic_decreasing_interval_f :
  ∀ a b, a > 2 ∧ b > 2 ∧ a < b →
  ∀ x y, x ∈ Set.Ioo a b ∧ y ∈ Set.Ioo a b ∧ x < y →
  f x > f y :=
by
  sorry

-- Additional lemma to show the domain of f
lemma domain_of_f (x : ℝ) :
  (x < 0 ∨ x > 2) ↔ x^2 - 2*x > 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_f_l588_58862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_function_properties_l588_58897

/-- A function satisfying the given property --/
def SpecialFunction (f : ℕ → ℕ) : Prop :=
  ∀ a b : ℕ, 2 * f (a^2 + b^2) = (f a)^2 + (f b)^2

/-- The set of possible values for f(18) --/
def PossibleValues (f : ℕ → ℕ) : Set ℕ :=
  {x : ℕ | SpecialFunction f ∧ f 18 = x}

theorem special_function_properties :
  ∃ (S : Finset ℕ), 
    (∀ f : ℕ → ℕ, SpecialFunction f → f 18 ∈ S) ∧ 
    S.card = 3 ∧
    S.sum id = 5 := by
  sorry

#check special_function_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_function_properties_l588_58897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_coverage_is_25_percent_l588_58833

/-- Represents a tiling of the plane with squares and hexagons -/
structure Tiling where
  /-- The side length of a small square -/
  small_square_side : ℝ
  /-- Assertion that the small square side length is positive -/
  small_square_side_pos : 0 < small_square_side

/-- The area of a large square in the tiling -/
def large_square_area (t : Tiling) : ℝ := 4 * t.small_square_side ^ 2

/-- The area of a small square in the tiling -/
def small_square_area (t : Tiling) : ℝ := t.small_square_side ^ 2

/-- The area of a hexagon in the tiling -/
def hexagon_area (t : Tiling) : ℝ := 4 * small_square_area t

/-- The fraction of a large square's area that is part of a hexagon -/
noncomputable def hexagon_fraction (t : Tiling) : ℝ := small_square_area t / large_square_area t

/-- Theorem stating that 25% of a large square's area is covered by hexagons -/
theorem hexagon_coverage_is_25_percent (t : Tiling) :
  hexagon_fraction t = 1/4 := by
  -- Unfold definitions
  unfold hexagon_fraction
  unfold small_square_area
  unfold large_square_area
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_coverage_is_25_percent_l588_58833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_collinear_l588_58827

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a truncated triangular pyramid -/
structure TruncatedPyramid where
  A : Point3D
  B : Point3D
  C : Point3D
  A₁ : Point3D
  B₁ : Point3D
  C₁ : Point3D

/-- Returns the point of intersection of three planes -/
noncomputable def intersectionOfPlanes (p1 p2 p3 : Point3D → Point3D → Point3D → Prop) : Point3D :=
  sorry

/-- Returns the point of intersection of three lines -/
noncomputable def intersectionOfLines (l1 l2 l3 : Point3D → Point3D → Prop) : Point3D :=
  sorry

/-- Returns the centroid (intersection of medians) of a triangle -/
noncomputable def centroid (p1 p2 p3 : Point3D) : Point3D :=
  sorry

/-- Checks if three points are collinear -/
def collinear (p1 p2 p3 : Point3D) : Prop :=
  sorry

/-- Main theorem: The intersection points lie on a single line -/
theorem intersection_points_collinear (pyramid : TruncatedPyramid) :
  let P := intersectionOfPlanes
    (λ a b c => sorry) -- Plane of triangle ABC₁
    (λ a b c => sorry) -- Plane of triangle BCA₁
    (λ a b c => sorry) -- Plane of triangle CAB₁
  let Q := intersectionOfLines
    (λ a b => sorry) -- Line AA₁
    (λ a b => sorry) -- Line BB₁
    (λ a b => sorry) -- Line CC₁
  let R := centroid pyramid.A pyramid.B pyramid.C
  let S := centroid pyramid.A₁ pyramid.B₁ pyramid.C₁
  collinear P Q R ∧ collinear P Q S :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_collinear_l588_58827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l588_58843

-- Define the function f
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (Real.sqrt 3 * Real.sin (2 * x) - 2 * (Real.cos x)^2 - a) / 2

-- Define the interval
def I : Set ℝ := Set.Icc (-Real.pi/12) (Real.pi/2)

-- State the theorem
theorem function_properties 
  (h_max : ∃ x ∈ I, f x (-1) = 2) 
  (α β : ℝ) 
  (h_α : α ∈ Set.Ioo 0 (Real.pi/2)) 
  (h_β : β ∈ Set.Ioo 0 (Real.pi/2))
  (h_fα : f (α/2 + Real.pi/12) (-1) = 10/13)
  (h_fβ : f (β/2 + Real.pi/3) (-1) = 6/5) :
  (∀ x ∈ I, -Real.sqrt 3 ≤ f x (-1) ∧ f x (-1) ≤ 2) ∧ 
  Real.sin (α - β) = -33/65 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l588_58843
